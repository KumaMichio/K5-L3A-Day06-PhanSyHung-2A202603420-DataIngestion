"""Day 06 pipeline entry point.

    python -m src.run prepare    # parse + normalize PDF, manifest, page log, chunk exports
    python -m src.run index      # embed A/B chunks, upsert into pgvector, reload check
    python -m src.run evaluate   # validate benchmark, top-3 for A/B/C, traces, metrics, audit sheets
    python -m src.run all        # all three in order
"""
import csv
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from . import config
from .chunking import by_paragraph, fixed_size
from .evaluation import evaluate_query, normalize_for_match, summarize
from .ingestion import build_corpus
from .models import Chunk, DocumentPage

PAGES_PATH = config.DATA_DIR / "pages.jsonl"
MANIFEST_PATH = config.DATA_DIR / "manifest.json"
CHUNK_DIR = config.RESULTS_DIR / "chunks"
TRACE_DIR = config.RESULTS_DIR / "traces"
AUDIT_DIR = config.RESULTS_DIR / "audit"
CACHE_DIR = config.STARTER_DIR / ".cache" / "fastembed"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _versions() -> dict[str, str]:
    packages = ["pymupdf", "psycopg", "fastembed", "onnxruntime", "python-dotenv"]
    return {"python": platform.python_version(), **{p: version(p) for p in packages}}


def load_pages() -> list[DocumentPage]:
    if not PAGES_PATH.exists():
        sys.exit("data/pages.jsonl missing: run `python -m src.run prepare` first")
    return [DocumentPage(**row) for row in _read_jsonl(PAGES_PATH)]


def build_chunks(pages: list[DocumentPage], index: str) -> list[Chunk]:
    params = next(p for p in config.STRATEGIES.values() if p["index"] == index)
    chunks: list[Chunk] = []
    for page in pages:
        if params["chunking"] == "fixed_size":
            chunks.extend(fixed_size(page, params["size"], params["overlap"]))
        else:
            chunks.extend(by_paragraph(page, params["max_size"]))
    return chunks


def _length_stats(lengths: list[int]) -> dict[str, Any]:
    ordered = sorted(lengths)
    return {"count": len(ordered), "total": sum(ordered), "min": ordered[0], "max": ordered[-1],
            "mean": round(statistics.mean(ordered), 1), "median": statistics.median(ordered),
            "p10": ordered[len(ordered) // 10], "p90": ordered[(len(ordered) * 9) // 10]}


# --------------------------------------------------------------------------- prepare
def prepare() -> None:
    started = time.perf_counter()
    pages, log = build_corpus(config.CORPUS, config.REPO_ROOT, config.MIN_PAGE_CHARS)
    parse_seconds = time.perf_counter() - started
    if len(pages) < config.MIN_CONTENT_PAGES:
        raise RuntimeError(f"only {len(pages)} content pages; need {config.MIN_CONTENT_PAGES}")
    _write_jsonl(PAGES_PATH, [asdict(p) for p in pages])

    manifest = {"generated_at": _now(), "counting_rule": (
        "A page counts when it is a physical PDF page inside the declared chapter ranges, is not a "
        "chapter-opener page that only holds a figure and chapter outline, is not an Assessment "
        "Questions / glossary / answer key / index page, and has at least "
        f"{config.MIN_PAGE_CHARS} characters after removing footers, footnotes, figure credits "
        "and outline blocks. Page IDs are 1-based physical PDF pages; printed page = PDF page - 8."),
        "total_content_pages": len(pages), "total_normalized_chars": sum(len(p.text) for p in pages),
        "sources": []}
    for spec, src_log in zip(config.CORPUS, log["sources"]):
        src_pages = [p for p in pages if p.document_id == spec["source_id"]]
        manifest["sources"].append({
            "source_id": spec["source_id"], "title": spec["title"], "authors": spec["authors"],
            "source_path_or_url": spec["pdf_url"], "landing_page": spec["source_url"],
            "local_path": spec["local_path"], "sha256": spec["sha256"],
            "license_or_permission": spec["license"], "accessed_at": spec["accessed_at"],
            "version_or_snapshot": f"OpenStax web PDF, publish_date {spec['publish_date']} (from OpenStax CMS API)",
            "physical_pages": src_log["physical_pages"],
            "chapters": spec["chapters"],
            "excluded_short_pages": src_log["excluded_short_pages"],
            "counted_pages": [{"page_id": p.page, "printed_page": p.metadata["printed_page"],
                               "chapter": p.metadata["chapter"], "chars": len(p.text)} for p in src_pages],
            "removed_blocks_total": src_log["removed_blocks"],
            "attribution": ("Content from Introduction to Intellectual Property by OpenStax, "
                            "https://openstax.org/details/books/introduction-intellectual-property, "
                            "licensed CC BY 4.0. Modified: text extracted, footers/footnotes removed, "
                            "whitespace normalized."),
        })
    _write_json(MANIFEST_PATH, manifest)

    page_log = [{"page": p.page, "printed_page": p.metadata["printed_page"], "chapter": p.metadata["chapter"],
                 "chars": len(p.text), "sections": [s["section"] for s in p.metadata["section_spans"]],
                 "removed_blocks": p.metadata["removed_blocks"]} for p in pages]
    _write_json(config.RESULTS_DIR / "ingestion_log.json", {
        "generated_at": _now(), "parse_seconds": round(parse_seconds, 2), "warnings": log["warnings"],
        "sources": log["sources"], "ocr": "not used: every counted page has a text layer",
        "pages": page_log})

    chunk_stats = {}
    for index in ("A", "B"):
        t0 = time.perf_counter()
        chunks = build_chunks(pages, index)
        chunk_seconds = time.perf_counter() - t0
        _write_jsonl(CHUNK_DIR / f"{index}.jsonl", [asdict(c) for c in chunks])
        chunk_stats[index] = {"chunk_seconds": round(chunk_seconds, 3),
                              "chars": _length_stats([len(c.text) for c in chunks]),
                              "chunks_missing_section": sum(c.metadata["section"] is None for c in chunks)}
    _write_json(config.RESULTS_DIR / "chunk_stats.json", chunk_stats)
    print(f"prepare: {len(pages)} pages, {manifest['total_normalized_chars']} chars, "
          f"A={chunk_stats['A']['chars']['count']} chunks, B={chunk_stats['B']['chars']['count']} chunks "
          f"({parse_seconds:.1f}s parse)")


# --------------------------------------------------------------------------- index
def _make_embedder():
    from .embeddings import FastEmbedEmbedder
    e = config.EMBEDDING
    return FastEmbedEmbedder(e["model_name"], e["dimensions"], e["query_prefix"], e["document_prefix"],
                             e["batch_size"], CACHE_DIR)


def _make_store():
    from .store import PgVectorStore
    load_dotenv(config.STARTER_DIR / ".env")
    url = os.getenv("DATABASE_URL")
    if not url:
        sys.exit("DATABASE_URL missing: copy starter/.env.example to starter/.env")
    store = PgVectorStore(url, config.EMBEDDING["dimensions"], config.EMBEDDING["model_name"],
                          config.FILTERABLE_FIELDS)
    store.create_schema()
    return store


def ingest_index(store, embedder, index: str, chunks: list[Chunk]) -> dict[str, Any]:
    """Embed only new/changed chunks, upsert them, delete obsolete ones; return counts + timings."""
    from .store import content_hash
    existing = store.existing_hashes(index)
    todo = [c for c in chunks if existing.get(c.chunk_id) != content_hash(c, embedder.model_name)]
    t0 = time.perf_counter()
    vectors = embedder.embed_documents([c.text for c in todo]) if todo else []
    embed_seconds = time.perf_counter() - t0
    t1 = time.perf_counter()
    counts = store.upsert(index, todo, vectors)
    deleted = store.delete_obsolete(index, {c.document_id for c in chunks}, {c.chunk_id for c in chunks})
    write_seconds = time.perf_counter() - t1
    counts.update({"skipped_unchanged_before_embedding": len(chunks) - len(todo), "deleted_obsolete": deleted,
                   "embedded": len(todo), "embed_seconds": round(embed_seconds, 2),
                   "write_seconds": round(write_seconds, 2)})
    return counts


def _token_stats(chunks: list[Chunk]) -> dict[str, Any] | None:
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_pretrained(config.EMBEDDING["model_name"])
    except Exception as exc:
        return {"error": f"tokenizer unavailable: {exc}"}
    lengths = [len(tok.encode(c.text).ids) for c in chunks]
    stats = _length_stats(lengths)
    stats["over_512_tokens"] = sum(n > 512 for n in lengths)
    return stats


def change_simulation(store, embedder, pages: list[DocumentPage]) -> dict[str, Any]:
    """Demonstrate changed-document handling on a scratch strategy, then clean it up."""
    scratch = "_change_test"
    page = next(p for p in pages if p.page == 116)
    original = by_paragraph(page, config.STRATEGIES["B"]["max_size"])
    first = ingest_index(store, embedder, scratch, original)
    paragraphs = page.text.split("\n\n")
    kept = paragraphs[: len(paragraphs) // 2]
    edited_text = "\n\n".join(kept[:-1] + [kept[-1] + " [edited sentence]"])
    edited_page = replace(page, text=edited_text)
    edited = by_paragraph(edited_page, config.STRATEGIES["B"]["max_size"])
    second = ingest_index(store, embedder, scratch, edited)
    rows_after = store.stats(scratch)
    store.delete_obsolete(scratch, {page.document_id}, set())
    return {"page": page.page, "first_ingest": first, "after_edit": second,
            "rows_after_edit": rows_after, "expected_rows_after_edit": len(edited),
            "note": "page 116 re-ingested after removing its second half and editing the last kept paragraph; scratch rows deleted afterwards"}


def index() -> None:
    pages = load_pages()
    store = _make_store()
    t0 = time.perf_counter()
    embedder = _make_embedder()
    load_seconds = time.perf_counter() - t0
    report: dict[str, Any] = {"generated_at": _now(), "versions": _versions(),
                              "store": {**store.server_versions(), "table": "chunks", "metric": "cosine (<=>)",
                                        "search": "exact (sequential scan, no ANN index)"},
                              "embedding": {**config.EMBEDDING, "onnx_artifact": embedder.onnx_source,
                                            "model_load_seconds": round(load_seconds, 2)},
                              "cost_usd": 0.0, "indexes": {}}
    for idx in ("A", "B"):
        chunks = build_chunks(pages, idx)
        first = ingest_index(store, embedder, idx, chunks)
        reload = ingest_index(store, embedder, idx, chunks)
        stats = store.stats(idx)
        ok = (stats["rows"] == len(chunks) == stats["distinct_chunk_ids"]
              and reload["inserted"] == reload["updated"] == reload["deleted_obsolete"] == 0)
        report["indexes"][idx] = {"chunks": len(chunks), "ingest": first, "reload_same_version": reload,
                                  "db_stats": stats, "reload_no_duplicates": ok,
                                  "tokens": _token_stats(chunks)}
        print(f"index {idx}: {len(chunks)} chunks, first={first}, reload_ok={ok}")
    report["change_simulation"] = change_simulation(store, embedder, pages)
    _write_json(config.RESULTS_DIR / "index_report.json", report)


# --------------------------------------------------------------------------- evaluate
REQUIRED_FIELDS = ("id", "split", "question", "expected_answer", "gold_source_id", "gold_page", "evidence_quote")


def load_benchmark(pages: list[DocumentPage]) -> list[dict[str, Any]]:
    data = json.loads(config.BENCHMARK_PATH.read_text(encoding="utf-8"))
    questions = data["questions"]
    page_text = {(p.document_id, p.page): normalize_for_match(p.text) for p in pages}
    errors = []
    for q in questions:
        missing = [f for f in REQUIRED_FIELDS if f not in q]
        if missing:
            errors.append(f"{q.get('id')}: missing {missing}")
            continue
        evidence = [(q["gold_source_id"], q["gold_page"], q["evidence_quote"])]
        evidence += [(a["source_id"], a["page"], a["evidence_quote"]) for a in q.get("alternative_evidence", [])]
        for source_id, page, quote in evidence:
            for pg in page if isinstance(page, list) else [page]:
                text = page_text.get((source_id, pg))
                if text is None:
                    errors.append(f"{q['id']}: page {pg} not in counted corpus")
                elif normalize_for_match(quote) not in text:
                    errors.append(f"{q['id']}: evidence not found on page {pg}")
        for field in (q.get("filter") or {}):
            if field not in config.FILTERABLE_FIELDS:
                errors.append(f"{q['id']}: filter field {field} not allowed")
    ids = [q["id"] for q in questions]
    dev = sum(q["split"] == "dev" for q in questions)
    test = sum(q["split"] == "test" for q in questions)
    filt = [q for q in questions if q.get("filter")]
    if len(ids) != len(set(ids)):
        errors.append("duplicate question IDs")
    if len(questions) < 20 or dev < 15 or test < 5:
        errors.append(f"need >=20 questions, >=15 dev, >=5 test (have {len(questions)}/{dev}/{test})")
    if len(filt) < 3 or not any(q["split"] == "test" for q in filt):
        errors.append("need >=3 filter questions including >=1 test question")
    if errors:
        raise ValueError("benchmark validation failed:\n  " + "\n  ".join(errors))
    return questions


def coherence_positions(pages: list[DocumentPage]) -> list[dict[str, Any]]:
    """10 positions fixed by rule before results: pages at evenly spaced ranks, start of 3rd paragraph."""
    positions = []
    for i in range(10):
        page = pages[(i * len(pages)) // 10]
        paragraphs = page.text.split("\n\n")
        para_index = min(2, len(paragraphs) - 1)
        offset = sum(len(p) + 2 for p in paragraphs[:para_index])
        positions.append({"position_id": f"pos{i + 1:02d}", "page": page.page, "offset": offset,
                          "paragraph_index": para_index, "preview": paragraphs[para_index][:120]})
    return positions


def write_audit_sheets(pages: list[DocumentPage], traces: list[dict[str, Any]],
                       questions: list[dict[str, Any]]) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    chunk_sets = {idx: [Chunk(**c) for c in _read_jsonl(CHUNK_DIR / f"{idx}.jsonl")] for idx in ("A", "B")}

    with (AUDIT_DIR / "coherence_audit.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["position_id", "page", "offset", "position_preview", "strategy", "chunk_id",
                    "chunk_chars", "chunk_text", "score_0_2", "reason"])
        for pos in coherence_positions(pages):
            for strategy in ("A", "B"):
                chunk = next(c for c in chunk_sets[strategy] if c.page == pos["page"]
                             and c.metadata["char_start"] <= pos["offset"] < c.metadata["char_end"])
                w.writerow([pos["position_id"], pos["page"], pos["offset"], pos["preview"],
                            strategy if strategy == "A" else "B (=C chunks)", chunk.chunk_id,
                            len(chunk.text), chunk.text, "", ""])

    test_traces = [t for t in traces if t["split"] == "test"]
    with (AUDIT_DIR / "metadata_audit.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "question_id", "question_filter", "rank", "chunk_id", "source", "page",
                    "printed_page", "chapter", "section", "subsection", "text_found_on_page_pypdf",
                    "chunk_text", "metadata_complete_and_correct", "error_notes"])
        independent = _independent_page_text()
        for t in test_traces:
            for h in t["hits"]:
                probe = normalize_for_match(h["text"])[:80].replace(" ", "")
                found = probe in independent.get(h["page"], "")
                w.writerow([t["strategy"], t["id"], json.dumps(t["question_filter"]) if t["question_filter"] else "",
                            h["rank"], h["chunk_id"], h["source"], h["page"], h["printed_page"], h["chapter"],
                            h["section"], h["subsection"], found, h["text"], "", ""])

    with (AUDIT_DIR / "grounding_audit.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "question_id", "question", "expected_answer", "gold_page", "evidence_quote",
                    "auto_evidence_hit_at_3", "top1_page", "top1_text", "top2_page", "top2_text",
                    "top3_page", "top3_text", "grounding_0_2", "reason"])
        for t in test_traces:
            q = next(q for q in questions if q["id"] == t["id"])
            row = [t["strategy"], t["id"], t["question"], t["expected_answer"], q["gold_page"],
                   q["evidence_quote"], t["hit_at_3"]]
            for h in t["hits"]:
                row += [h["page"], h["text"]]
            w.writerow(row + ["", ""])


def _independent_page_text() -> dict[int, str]:
    """Second extractor (pypdf) used only to cross-check page metadata in the audit."""
    from pypdf import PdfReader
    spec = config.CORPUS[0]
    reader = PdfReader(config.REPO_ROOT / spec["local_path"])
    wanted = {p["page_id"] for p in json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["sources"][0]["counted_pages"]}
    return {i + 1: normalize_for_match(reader.pages[i].extract_text() or "").replace(" ", "")
            for i in range(len(reader.pages)) if i + 1 in wanted}


def evaluate() -> None:
    pages = load_pages()
    questions = load_benchmark(pages)
    store = _make_store()
    embedder = _make_embedder()
    run_at = _now()
    traces: list[dict[str, Any]] = []
    timings: dict[str, float] = {}
    query_vectors = {q["id"]: embedder.embed_query(q["question"]) for q in questions}
    for strategy, params in config.STRATEGIES.items():
        t0 = time.perf_counter()
        for q in questions:
            applied = q.get("filter") if params["use_filter"] and q.get("filter") else None
            hits = store.search(params["index"], query_vectors[q["id"]], config.TOP_K, applied)
            if len(hits) < config.TOP_K:
                raise RuntimeError(f"{strategy}/{q['id']}: only {len(hits)} hits")
            traces.append(evaluate_query(q, hits, strategy, applied))
        timings[strategy] = round(time.perf_counter() - t0, 3)
        _write_jsonl(TRACE_DIR / f"{strategy}.jsonl", [t for t in traces if t["strategy"] == strategy])
    summary = summarize(traces)
    _write_json(config.RESULTS_DIR / "metrics.json", {
        "run_at": run_at, "top_k": config.TOP_K, "embedding_model": config.EMBEDDING["model_name"],
        "strategies": config.STRATEGIES, "search_seconds_all_questions": timings,
        "questions": {"total": len(questions), "dev": sum(q["split"] == "dev" for q in questions),
                      "test": sum(q["split"] == "test" for q in questions)},
        "summary": summary})
    write_audit_sheets(pages, traces, questions)
    for strategy, by_split in summary.items():
        for split in ("dev", "test"):
            s = by_split[split]
            print(f"{strategy} {split}: Hit@1 {s['hit_at_1']['n']}/{s['hit_at_1']['N']}  "
                  f"Hit@3 {s['evidence_hit_at_3']['n']}/{s['evidence_hit_at_3']['N']}  "
                  f"filter compliance {s['filter_compliance_top3']}")


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    stages = {"prepare": [prepare], "index": [index], "evaluate": [evaluate],
              "all": [prepare, index, evaluate]}
    if stage not in stages:
        sys.exit(f"unknown stage {stage!r}; choose from {', '.join(stages)}")
    for fn in stages[stage]:
        fn()


if __name__ == "__main__":
    main()

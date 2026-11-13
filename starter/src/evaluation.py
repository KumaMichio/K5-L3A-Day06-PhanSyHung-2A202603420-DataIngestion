"""Evaluate retrieved evidence (Hit@1, Evidence Hit@3) and keep full per-query traces.

Coherence and grounding are human labels; nothing here infers them from string match.
"""
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any

QUOTE_MAP = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "—": "-", "–": "-"})
SPLIT_SUSPECT_COVERAGE = 0.5


def normalize_for_match(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(QUOTE_MAP).lower()
    return re.sub(r"\s+", " ", text).strip()


def gold_evidence(question: dict[str, Any]) -> list[dict[str, Any]]:
    """Primary gold plus any documented alternative evidence (all are accepted)."""
    gold = [{"source_id": question["gold_source_id"],
             "pages": question["gold_page"] if isinstance(question["gold_page"], list) else [question["gold_page"]],
             "evidence_quote": question["evidence_quote"]}]
    for alt in question.get("alternative_evidence", []):
        pages = alt["page"] if isinstance(alt["page"], list) else [alt["page"]]
        gold.append({"source_id": alt["source_id"], "pages": pages,
                     "evidence_quote": alt["evidence_quote"]})
    return gold


def _coverage(quote: str, text: str) -> float:
    """Share of the quote covered by its longest contiguous match in the chunk."""
    if not quote:
        return 0.0
    match = SequenceMatcher(None, quote, text, autojunk=False).find_longest_match(
        0, len(quote), 0, len(text))
    return match.size / len(quote)


def _filter_ok(hit: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    return all(hit.get(k) == v for k, v in (filters or {}).items())


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]],
                   strategy: str, filter_applied: dict[str, Any] | None) -> dict[str, Any]:
    gold = gold_evidence(question)
    ranked = []
    for rank, hit in enumerate(hits, start=1):
        text_norm = normalize_for_match(hit["text"])
        on_gold_page = any(hit["document_id"] == g["source_id"] and hit["page"] in g["pages"] for g in gold)
        is_hit = any(hit["document_id"] == g["source_id"] and hit["page"] in g["pages"]
                     and normalize_for_match(g["evidence_quote"]) in text_norm for g in gold)
        coverage = max((_coverage(normalize_for_match(g["evidence_quote"]), text_norm)
                        for g in gold if hit["document_id"] == g["source_id"] and hit["page"] in g["pages"]),
                       default=0.0)
        ranked.append({
            "rank": rank, "chunk_id": hit["chunk_id"], "score": round(float(hit["score"]), 6),
            "document_id": hit["document_id"], "source": hit["source"], "page": hit["page"],
            "printed_page": hit["printed_page"], "chapter": hit["chapter"],
            "chapter_title": hit["chapter_title"], "section": hit["section"],
            "subsection": hit["subsection"], "text": hit["text"],
            "on_gold_page": on_gold_page, "evidence_coverage": round(coverage, 3),
            "is_evidence_hit": is_hit,
            "matches_question_filter": _filter_ok(hit, question.get("filter")) if question.get("filter") else None,
        })
    first_hit = next((r["rank"] for r in ranked if r["is_evidence_hit"]), None)
    split_suspected = first_hit is None and any(
        r["on_gold_page"] and r["evidence_coverage"] >= SPLIT_SUSPECT_COVERAGE for r in ranked)
    return {
        "strategy": strategy,
        "id": question["id"],
        "split": question["split"],
        "type": question.get("type"),
        "question": question["question"],
        "expected_answer": question["expected_answer"],
        "question_filter": question.get("filter"),
        "filter_applied": filter_applied,
        "gold": gold,
        "hits": ranked,
        "hit_at_1": first_hit == 1,
        "hit_at_3": first_hit is not None,
        "first_hit_rank": first_hit,
        "gold_page_in_top3": any(r["on_gold_page"] for r in ranked),
        "evidence_split_suspected": split_suspected,
        "top3_respects_filter": (all(r["matches_question_filter"] for r in ranked)
                                 if question.get("filter") else None),
    }


def _rate(n: int, d: int) -> dict[str, Any]:
    return {"n": n, "N": d, "rate": round(n / d, 3) if d else None}


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for strategy in sorted({t["strategy"] for t in traces}):
        summary[strategy] = {}
        for split in ("dev", "test", "all"):
            rows = [t for t in traces if t["strategy"] == strategy and (split == "all" or t["split"] == split)]
            filtered = [t for t in rows if t["question_filter"]]
            summary[strategy][split] = {
                "hit_at_1": _rate(sum(t["hit_at_1"] for t in rows), len(rows)),
                "evidence_hit_at_3": _rate(sum(t["hit_at_3"] for t in rows), len(rows)),
                "gold_page_in_top3": _rate(sum(t["gold_page_in_top3"] for t in rows), len(rows)),
                "evidence_split_suspected": sum(t["evidence_split_suspected"] for t in rows),
                "filter_questions_hit_at_3": _rate(sum(t["hit_at_3"] for t in filtered), len(filtered)),
                "filter_compliance_top3": _rate(sum(bool(t["top3_respects_filter"]) for t in filtered),
                                                len(filtered)) if filtered else "N/A",
            }
    return summary

"""Read and normalize source data with traceability back to PDF page and section."""
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import pymupdf

from .models import DocumentPage

BODY_MIN_SIZE = 8.0
CHAPTER_TITLE_MIN_SIZE = 20.0
SUBHEADING_MIN_SIZE = 12.0
SUPERSCRIPT_FLAG = 1
BOLD_FLAG = 16

FOOTER_PATTERNS = [
    re.compile(r"^Access for free at openstax\.org$"),
    re.compile(r"^\d+\s+\d+\s*•.*$"),
    re.compile(r"^.*•.*\s+\d+$"),
]
FIGURE_CREDIT = re.compile(r"^Figure \d+\.\d+ \(credit")
FOOTNOTE_MARKER = re.compile(r"^[ivxlcdm]+$")
LIST_ITEM =re.compile(r"^(•|\d+\.\s|[a-z]\.\s)")
NON_SECTION_TOC = {"Chapter Outline", "Assessment Questions"}


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace while keeping paragraph and list breaks."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("­", "").replace("​", "").replace("﻿", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    lines = [re.sub(r" {2,}", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _line_text(line: dict[str, Any]) -> tuple[str, int]:
    """Join a line's spans, dropping superscript footnote markers."""
    parts, dropped = [], 0
    for span in line["spans"]:
        if (span["flags"] & SUPERSCRIPT_FLAG and span["size"] < BODY_MIN_SIZE
                and FOOTNOTE_MARKER.match(span["text"].strip())):
            dropped += 1
            continue
        parts.append(span["text"])
    return "".join(parts), dropped


def _block_text(block: dict[str, Any]) -> tuple[str, int]:
    out, dropped = "", 0
    for line in block.get("lines", []):
        text, n = _line_text(line)
        dropped += n
        text = text.strip()
        if not text:
            continue
        if not out:
            out = text
        elif LIST_ITEM.match(text):
            out += "\n" + text
        elif out.endswith("-") and len(out) > 1 and out[-2].isalpha():
            out += text
        else:
            out += " " + text
    return out, dropped


def _block_style(block: dict[str, Any]) -> tuple[float, bool]:
    spans = [s for line in block.get("lines", []) for s in line["spans"] if s["text"].strip()]
    if not spans:
        return 0.0, False
    size = max(s["size"] for s in spans)
    bold = all(s["flags"] & BOLD_FLAG for s in spans)
    return size, bold


def _toc_index(doc: pymupdf.Document) -> tuple[dict[int, list[str]], list[tuple[int, int, str]]]:
    """Map page -> section titles starting there, and list (page, chapter_no, chapter_title)."""
    sections: dict[int, list[str]] = {}
    chapters: list[tuple[int, int, str]] = []
    for level, title, page in doc.get_toc():
        title = normalize_text(title)
        if level == 1:
            match = re.match(r"Chapter (\d+) (.+)", title)
            if match:
                chapters.append((page, int(match.group(1)), match.group(2)))
        elif level == 2:
            sections.setdefault(page, []).append(title)
    return sections, chapters


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Extract every physical page with page number, section spans and a cleaning log.

    Pages with no text after cleaning are still returned (text == "") so the caller
    can log them; OCR is not attempted because this PDF has a text layer.
    """
    doc = pymupdf.open(path)
    toc_sections, toc_chapters = _toc_index(doc)
    pages: list[DocumentPage] = []
    section: str | None = None
    subsection: str | None = None
    chapter_no: int | None = None
    chapter_title: str | None = None

    for index, pdf_page in enumerate(doc):
        page_no = index + 1
        for start, number, title in toc_chapters:
            if start == page_no:
                chapter_no, chapter_title, section, subsection = number, title, None, None
        starting_here = set(toc_sections.get(page_no, []))
        removed: Counter[str] = Counter()
        paragraphs: list[str] = []
        spans: list[dict[str, Any]] = []
        offset = 0
        in_outline = False
        page_dict = pdf_page.get_text("dict")

        for block in page_dict["blocks"]:
            if block.get("type") != 0:
                removed["image"] += 1
                continue
            text, dropped = _block_text(block)
            removed["superscript_marker"] += dropped
            text = normalize_text(text)
            if not text:
                removed["empty"] += 1
                continue
            size, bold = _block_style(block)
            flat = text.replace("\n", " ")

            if size < BODY_MIN_SIZE:
                kind = "footer" if bold and any(p.match(flat) for p in FOOTER_PATTERNS) else "footnote"
                removed[kind] += 1
                continue
            if FIGURE_CREDIT.match(flat):
                removed["figure_credit"] += 1
                continue
            if size >= CHAPTER_TITLE_MIN_SIZE or flat == f"{chapter_title} {chapter_no}":
                removed["chapter_title"] += 1
                continue
            if flat == "Chapter Outline":
                in_outline = True
                removed["chapter_outline"] += 1
                continue
            if in_outline:
                if flat in starting_here and flat not in NON_SECTION_TOC:
                    in_outline = False
                else:
                    removed["chapter_outline"] += 1
                    continue

            if flat in starting_here and flat not in NON_SECTION_TOC:
                section, subsection = flat, None
                is_heading = True
            elif bold and size >= SUBHEADING_MIN_SIZE and len(flat) < 120:
                subsection = flat
                is_heading = True
            else:
                is_heading = False

            if is_heading or not spans or spans[-1]["section"] != section or spans[-1]["subsection"] != subsection:
                spans.append({"offset": offset, "section": section, "subsection": subsection})
            paragraphs.append(text)
            offset += len(text) + 2

        page_text = "\n\n".join(paragraphs)
        pages.append(DocumentPage(
            document_id=document_id,
            source=source,
            page=page_no,
            text=page_text,
            metadata={
                "chapter": chapter_no,
                "chapter_title": chapter_title,
                "section_spans": spans,
                "removed_blocks": dict(removed),
                "char_count": len(page_text),
            },
        ))
    doc.close()
    return pages


def section_at(page: DocumentPage, offset: int) -> tuple[str | None, str | None]:
    """Return (section, subsection) active at a character offset of the page text."""
    current = (None, None)
    for span in page.metadata.get("section_spans", []):
        if span["offset"] <= offset:
            current = (span["section"], span["subsection"])
        else:
            break
    return current


def build_corpus(corpus_spec: list[dict[str, Any]], repo_root: Path,
                 min_page_chars: int) -> tuple[list[DocumentPage], dict[str, Any]]:
    """Load every source, keep only declared content pages, and return a processing log."""
    selected: list[DocumentPage] = []
    log: dict[str, Any] = {"sources": [], "warnings": []}
    for spec in corpus_spec:
        path = repo_root / spec["local_path"]
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run `python scripts/download_corpus.py` first")
        all_pages = load_pdf(path, spec["source_id"], spec["local_path"])
        source_log = {"source_id": spec["source_id"], "physical_pages": len(all_pages),
                      "counted_pages": [], "excluded_short_pages": [], "removed_blocks": Counter()}
        for chapter in spec["chapters"]:
            first, last = chapter["pages"]
            for page in all_pages[first - 1:last]:
                if page.metadata["chapter"] != chapter["chapter"]:
                    raise ValueError(f"page {page.page} is chapter {page.metadata['chapter']}, "
                                     f"spec says {chapter['chapter']}")
                if page.metadata["chapter_title"] != chapter["chapter_title"]:
                    log["warnings"].append(f"chapter title mismatch on page {page.page}")
                source_log["removed_blocks"].update(page.metadata["removed_blocks"])
                if len(page.text) < min_page_chars:
                    source_log["excluded_short_pages"].append({"page": page.page, "chars": len(page.text)})
                    continue
                page.metadata["printed_page"] = page.page - spec["printed_page_offset"]
                source_log["counted_pages"].append(page.page)
                selected.append(page)
        source_log["removed_blocks"] = dict(source_log["removed_blocks"])
        log["sources"].append(source_log)
    return selected, log

"""Fixed-size and structure-aware chunking; every chunk keeps source/page/section."""
import re

from .ingestion import section_at
from .models import Chunk, DocumentPage

MIN_STANDALONE = 80
SENTENCE_END =re.compile(r"(?<=[.!?”\"])\s+(?=[A-Z“\"(•\d])")


def _make_chunk(page: DocumentPage, kind: str, index: int, start: int, end: int,
                meta_offset: int | None = None) -> Chunk:
    text = page.text[start:end].strip()
    section, subsection = section_at(page, start if meta_offset is None else meta_offset)
    return Chunk(
        chunk_id=f"{page.document_id}:p{page.page:03d}:{kind}:{index:02d}",
        document_id=page.document_id,
        source=page.source,
        page=page.page,
        text=text,
        metadata={
            "chapter": page.metadata["chapter"],
            "chapter_title": page.metadata["chapter_title"],
            "section": section,
            "subsection": subsection,
            "printed_page": page.metadata.get("printed_page"),
            "char_start": start,
            "char_end": end,
            "chunker": kind,
        },
    )


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Split page text into fixed character windows with overlap (no boundary snapping)."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("require size > 0 and 0 <= overlap < size")
    chunks: list[Chunk] = []
    step = size - overlap
    start = 0
    while start < len(page.text):
        end = min(start + size, len(page.text))
        if page.text[start:end].strip():
            chunks.append(_make_chunk(page, "fixed", len(chunks), start, end))
        if end == len(page.text):
            break
        start += step
    return chunks


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for match in re.finditer(r"\n\n", text):
        spans.append((start, match.start()))
        start = match.end()
    spans.append((start, len(text)))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def _split_oversized(text: str, start: int, end: int, max_size: int) -> list[tuple[int, int]]:
    """Split one long paragraph at sentence ends; fall back to word boundaries."""
    cuts = [start] + [start + m.end() for m in SENTENCE_END.finditer(text[start:end])] + [end]
    pieces: list[tuple[int, int]] = []
    piece_start = start
    for prev, cut in zip(cuts, cuts[1:]):
        if cut - piece_start > max_size and prev > piece_start:
            pieces.append((piece_start, prev))
            piece_start = prev
    pieces.append((piece_start, end))
    result: list[tuple[int, int]] = []
    for s, e in pieces:
        while e - s > max_size:
            cut = text.rfind(" ", s, s + max_size)
            cut = cut if cut > s else s + max_size
            result.append((s, cut))
            s = cut
        result.append((s, e))
    return result


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Group whole paragraphs up to max_size; start a new chunk at every heading."""
    heading_offsets = {span["offset"] for span in page.metadata.get("section_spans", [])}
    units: list[tuple[int, int, bool]] = []
    for s, e in _paragraph_spans(page.text):
        is_boundary = s in heading_offsets
        if e - s > max_size:
            for i, (ps, pe) in enumerate(_split_oversized(page.text, s, e, max_size)):
                units.append((ps, pe, is_boundary and i == 0))
        else:
            units.append((s, e, is_boundary))

    groups: list[list[int]] = []
    for s, e, boundary in units:
        if not groups:
            groups.append([s, e, s])
            continue
        g_start, g_end, _ = groups[-1]
        heading_only = g_end - g_start < MIN_STANDALONE
        if heading_only and e - g_start <= max_size:
            groups[-1] = [g_start, e, s if boundary else groups[-1][2]]
        elif not boundary and e - g_start <= max_size:
            groups[-1][1] = e
        else:
            groups.append([s, e, s])
    return [_make_chunk(page, "para", i, s, e, meta_offset=m) for i, (s, e, m) in enumerate(groups)]

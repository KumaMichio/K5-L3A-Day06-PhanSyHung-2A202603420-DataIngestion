"""Single source of truth for corpus, model and strategy parameters.

Every strategy reads the same corpus, embedding model and benchmark from here, so
the only differences between A, B and C are the ones declared in STRATEGIES.
"""
from pathlib import Path

STARTER_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = STARTER_DIR.parent

RAW_DIR = STARTER_DIR / "data" / "raw"
DATA_DIR = REPO_ROOT / "data"
BENCHMARK_PATH = REPO_ROOT / "benchmark" / "questions.json"
RESULTS_DIR = REPO_ROOT / "results"

CORPUS = [
    {
        "source_id": "openstax-intro-ip",
        "title": "Introduction to Intellectual Property (OpenStax)",
        "authors": "OpenStax / Openstax Intellectual Property Team",
        "publish_date": "2021-02-22",
        "source_url": "https://openstax.org/details/books/introduction-intellectual-property",
        "pdf_url": "https://assets.openstax.org/oscms-prodcms/media/documents/introduction-intellectual-property_-_WEB.pdf",
        "local_path": "starter/data/raw/introduction-intellectual-property.pdf",
        "sha256": "1498509864D37CB63A6CCC11BAB7C783F3D30B2ED37D808EF80C56FDB676378A",
        "license": "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)",
        "accessed_at": "2026-11-13",
        "printed_page_offset": 8,
        # Physical PDF pages (1-based) counted as content. Chapter-opener pages that
        # only hold a figure and outline, and the Assessment Questions pages, are excluded.
        "chapters": [
            {"chapter": 3, "chapter_title": "Copyright Basics", "pages": [102, 134]},
            {"chapter": 4, "chapter_title": "Trademark Basics", "pages": [141, 164]},
            {"chapter": 5, "chapter_title": "Trade Secret Basics", "pages": [172, 187]},
        ],
    }
]

MIN_CONTENT_PAGES = 30
MIN_PAGE_CHARS = 200

EMBEDDING = {
    "provider": "local (fastembed + ONNX Runtime, CPU)",
    "model_name": "BAAI/bge-small-en-v1.5",
    "dimensions": 384,
    "query_prefix": "Represent this sentence for searching relevant passages: ",
    "document_prefix": "",
    "normalize": True,
    "batch_size": 32,
}

TOP_K = 3

# C reuses B's index on purpose: same chunks and vectors, only the filter changes.
STRATEGIES = {
    "A": {"chunking": "fixed_size", "size": 500, "overlap": 50, "index": "A", "use_filter": False},
    "B": {"chunking": "by_paragraph", "max_size": 800, "index": "B", "use_filter": False},
    "C": {"chunking": "by_paragraph", "max_size": 800, "index": "B", "use_filter": True},
}

FILTERABLE_FIELDS = {"chapter", "section"}

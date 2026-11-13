"""Real local embedding model via fastembed (ONNX Runtime); no mock fallback."""
import math
from pathlib import Path
from typing import Protocol

from fastembed import TextEmbedding


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class FastEmbedEmbedder:
    """BGE-small via fastembed. Any load or inference error propagates to the caller."""

    def __init__(self, model_name: str, dimensions: int, query_prefix: str = "",
                 document_prefix: str = "", batch_size: int = 32,
                 cache_dir: Path | None = None) -> None:
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.batch_size = batch_size
        self._model = TextEmbedding(model_name, cache_dir=str(cache_dir) if cache_dir else None)
        probe = self._encode(["dimension probe"])[0]
        if len(probe) != dimensions:
            raise ValueError(f"{model_name} returned {len(probe)} dims, config says {dimensions}")
        self.dimensions = dimensions
        description = next((m for m in TextEmbedding.list_supported_models()
                            if m["model"] == model_name), {})
        self.onnx_source = description.get("sources", {}).get("hf")

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for vector in self._model.embed(texts, batch_size=self.batch_size):
            values = [float(x) for x in vector]
            norm = math.sqrt(sum(x * x for x in values))
            if norm == 0:
                raise ValueError("embedding model returned a zero vector")
            vectors.append([x / norm for x in values])
        if len(vectors) != len(texts):
            raise RuntimeError("embedding count does not match input count")
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode([self.document_prefix + t for t in texts])

    def embed_query(self, question: str) -> list[float]:
        return self._encode([self.query_prefix + question])[0]

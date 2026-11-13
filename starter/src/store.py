"""Store and search vectors with PostgreSQL + pgvector (exact cosine search)."""
import hashlib
import json
from typing import Any

import psycopg

from .models import Chunk

COLUMN_FIELDS = ("printed_page", "chapter", "chapter_title", "section", "subsection")


def content_hash(chunk: Chunk, model_name: str) -> str:
    payload = json.dumps({"text": chunk.text, "source": chunk.source, "page": chunk.page,
                          "metadata": chunk.metadata, "model": model_name},
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vector) + "]"


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int, model_name: str,
                 filterable_fields: set[str]) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        unknown = filterable_fields - set(COLUMN_FIELDS)
        if unknown:
            raise ValueError(f"filterable fields must be stored columns: {unknown}")
        self.database_url = database_url
        self.dimensions = dimensions
        self.model_name = model_name
        self.filterable_fields = filterable_fields

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self.database_url, connect_timeout=5)

    def server_versions(self) -> dict[str, str]:
        with self._connect() as conn:
            row = conn.execute("SELECT current_setting('server_version'), extversion "
                               "FROM pg_extension WHERE extname = 'vector'").fetchone()
        if not row:
            raise RuntimeError("pgvector extension is not installed")
        return {"postgres": row[0], "pgvector": row[1]}

    def create_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    strategy        text        NOT NULL,
                    chunk_id        text        NOT NULL,
                    document_id     text        NOT NULL,
                    source          text        NOT NULL,
                    page            integer     NOT NULL,
                    printed_page    integer,
                    chapter         integer,
                    chapter_title   text,
                    section         text,
                    subsection      text,
                    text            text        NOT NULL,
                    char_count      integer     NOT NULL,
                    content_hash    text        NOT NULL,
                    embedding_model text        NOT NULL,
                    embedding       vector({self.dimensions}) NOT NULL,
                    metadata        jsonb       NOT NULL,
                    ingested_at     timestamptz NOT NULL DEFAULT now(),
                    PRIMARY KEY (strategy, chunk_id)
                )""")
            conn.execute("CREATE INDEX IF NOT EXISTS chunks_strategy_chapter_idx "
                         "ON chunks (strategy, chapter)")
            dims = conn.execute("""
                SELECT atttypmod FROM pg_attribute
                WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'""").fetchone()[0]
            if dims != self.dimensions:
                raise RuntimeError(f"existing chunks.embedding has {dims} dims, model has "
                                   f"{self.dimensions}; use a new table instead of mixing models")

    def existing_hashes(self, strategy: str) -> dict[str, str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT chunk_id, content_hash FROM chunks WHERE strategy = %s",
                                (strategy,)).fetchall()
        return dict(rows)

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> dict[str, int]:
        """Insert new chunk IDs, update rows whose content hash changed, skip identical rows.

        Obsolete chunks of re-ingested documents are removed by `delete_obsolete`.
        """
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        for vector in vectors:
            if len(vector) != self.dimensions:
                raise ValueError(f"vector has {len(vector)} dims, expected {self.dimensions}")
        ids = [c.chunk_id for c in chunks]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate chunk IDs in one batch")

        inserted = updated = 0
        with self._connect() as conn, conn.transaction():
            for chunk, vector in zip(chunks, vectors):
                meta = chunk.metadata
                row = conn.execute("""
                    INSERT INTO chunks (strategy, chunk_id, document_id, source, page, printed_page,
                        chapter, chapter_title, section, subsection, text, char_count,
                        content_hash, embedding_model, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s)
                    ON CONFLICT (strategy, chunk_id) DO UPDATE SET
                        document_id = EXCLUDED.document_id, source = EXCLUDED.source,
                        page = EXCLUDED.page, printed_page = EXCLUDED.printed_page,
                        chapter = EXCLUDED.chapter, chapter_title = EXCLUDED.chapter_title,
                        section = EXCLUDED.section, subsection = EXCLUDED.subsection,
                        text = EXCLUDED.text, char_count = EXCLUDED.char_count,
                        content_hash = EXCLUDED.content_hash,
                        embedding_model = EXCLUDED.embedding_model,
                        embedding = EXCLUDED.embedding, metadata = EXCLUDED.metadata,
                        ingested_at = now()
                    WHERE chunks.content_hash <> EXCLUDED.content_hash
                    RETURNING (xmax = 0) AS inserted""",
                    (strategy, chunk.chunk_id, chunk.document_id, chunk.source, chunk.page,
                     meta.get("printed_page"), meta.get("chapter"), meta.get("chapter_title"),
                     meta.get("section"), meta.get("subsection"), chunk.text, len(chunk.text),
                     content_hash(chunk, self.model_name), self.model_name,
                     _vector_literal(vector), json.dumps(meta, ensure_ascii=False))).fetchone()
                if row is not None:
                    inserted += int(row[0])
                    updated += int(not row[0])
        return {"inserted": inserted, "updated": updated,
                "unchanged": len(chunks) - inserted - updated}

    def delete_obsolete(self, strategy: str, document_ids: set[str], keep_ids: set[str]) -> int:
        with self._connect() as conn, conn.transaction():
            cur = conn.execute("""
                DELETE FROM chunks
                WHERE strategy = %s AND document_id = ANY(%s) AND NOT (chunk_id = ANY(%s))""",
                (strategy, list(document_ids), list(keep_ids)))
            return cur.rowcount

    def stats(self, strategy: str) -> dict[str, Any]:
        with self._connect() as conn:
            total, distinct_ids, distinct_content = conn.execute("""
                SELECT count(*), count(DISTINCT chunk_id),
                       count(DISTINCT (document_id, page, md5(text)))
                FROM chunks WHERE strategy = %s""", (strategy,)).fetchone()
        return {"rows": total, "distinct_chunk_ids": distinct_ids,
                "distinct_doc_page_text": distinct_content}

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if len(query_vector) != self.dimensions:
            raise ValueError("query vector dimension mismatch")
        where, params = ["strategy = %s"], [strategy]
        for field, value in (filters or {}).items():
            if field not in self.filterable_fields:
                raise ValueError(f"filter on {field!r} is not permitted")
            where.append(f"{field} = %s")
            params.append(value)
        vector = _vector_literal(query_vector)
        sql = f"""
            SELECT chunk_id, document_id, source, page, printed_page, chapter, chapter_title,
                   section, subsection, text, 1 - (embedding <=> %s::vector) AS score
            FROM chunks WHERE {' AND '.join(where)}
            ORDER BY embedding <=> %s::vector, chunk_id
            LIMIT %s"""
        with self._connect() as conn:
            rows = conn.execute(sql, [vector, *params, vector, top_k]).fetchall()
        keys = ("chunk_id", "document_id", "source", "page", "printed_page", "chapter",
                "chapter_title", "section", "subsection", "text", "score")
        return [dict(zip(keys, row)) for row in rows]

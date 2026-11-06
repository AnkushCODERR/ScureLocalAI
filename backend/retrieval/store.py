from __future__ import annotations

import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.config import Settings, ensure_storage_dirs, get_settings


class ChromaStore:
    def __init__(self, settings: Settings | None = None) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.settings = settings or get_settings()
        ensure_storage_dirs(self.settings)
        self.client = chromadb.PersistentClient(
            path=str(self.settings.storage.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.settings.storage.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        doc_id: str,
        filename: str,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        ingested_at = datetime.now(timezone.utc).isoformat()
        ids = [f"{doc_id}:{chunk.get('chunk_index', idx)}" for idx, chunk in enumerate(chunks)]
        documents = [str(chunk.get("text", "")) for chunk in chunks]
        metadatas = [
            {
                "doc_id": doc_id,
                "filename": filename,
                "page": int(chunk.get("page", 1)),
                "chunk_index": int(chunk.get("chunk_index", idx)),
                "ingested_at": ingested_at,
                "is_image": bool(chunk.get("is_image", False)),
                "figure_id": str(chunk.get("figure_id", "")),
                "image_url": str(chunk.get("image_url", "")),
                "caption": str(chunk.get("caption", "")),
                "kind": str(chunk.get("kind", "")),
                "vlm_description": str(chunk.get("vlm_description", "")),
                "surrounding_text": str(chunk.get("surrounding_text", "")),
            }
            for idx, chunk in enumerate(chunks)
        ]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(self, embedding: list[float], k: int) -> list[dict[str, Any]]:
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=max(1, k),
            include=["documents", "metadatas", "distances"],
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        chunks: list[dict[str, Any]] = []
        for text, metadata, distance in zip(documents, metadatas, distances):
            metadata = metadata or {}
            chunks.append(
                {
                    "text": text,
                    "filename": metadata.get("filename", ""),
                    "doc_id": metadata.get("doc_id", ""),
                    "page": metadata.get("page"),
                    "chunk_index": metadata.get("chunk_index"),
                    "distance": distance,
                    "is_image": bool(metadata.get("is_image", False)),
                    "figure_id": metadata.get("figure_id", ""),
                    "image_url": metadata.get("image_url", ""),
                    "caption": metadata.get("caption", ""),
                    "kind": metadata.get("kind", ""),
                    "vlm_description": metadata.get("vlm_description", ""),
                    "surrounding_text": metadata.get("surrounding_text", ""),
                }
            )
        return chunks

    def delete_document(self, doc_id: str) -> None:
        self.collection.delete(where={"doc_id": doc_id})
        # Delete document figures directory if present
        figures_doc_dir = self.settings.vision.output_dir / doc_id
        if figures_doc_dir.exists():
            shutil.rmtree(figures_doc_dir, ignore_errors=True)

    def list_documents(self) -> list[dict[str, Any]]:
        result = self.collection.get(include=["metadatas"])
        documents: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"id": "", "filename": "", "ingested_at": "", "chunks": 0, "figures": 0}
        )

        for metadata in result.get("metadatas", []):
            if not metadata:
                continue
            doc_id = str(metadata.get("doc_id", ""))
            if not doc_id:
                continue
            record = documents[doc_id]
            record["id"] = doc_id
            record["filename"] = metadata.get("filename", "")
            record["ingested_at"] = metadata.get("ingested_at", "")
            record["chunks"] += 1
            if metadata.get("is_image"):
                record["figures"] += 1

        return sorted(documents.values(), key=lambda item: str(item["ingested_at"]), reverse=True)

    def list_figures(self, doc_id: str | None = None) -> list[dict[str, Any]]:
        where_filter = {"is_image": True} if not doc_id else {"$and": [{"doc_id": doc_id}, {"is_image": True}]}
        try:
            result = self.collection.get(where=where_filter, include=["metadatas", "documents"])
        except Exception:
            # Fallback filter in memory
            result = self.collection.get(include=["metadatas", "documents"])

        figures: list[dict[str, Any]] = []
        for text, metadata in zip(result.get("documents", []), result.get("metadatas", [])):
            if not metadata or not metadata.get("is_image"):
                continue
            if doc_id and metadata.get("doc_id") != doc_id:
                continue
            figures.append(
                {
                    "doc_id": metadata.get("doc_id", ""),
                    "filename": metadata.get("filename", ""),
                    "figure_id": metadata.get("figure_id", ""),
                    "page": metadata.get("page", 1),
                    "kind": metadata.get("kind", "unknown"),
                    "caption": metadata.get("caption", ""),
                    "image_url": metadata.get("image_url", ""),
                    "vlm_description": metadata.get("vlm_description", ""),
                    "surrounding_text": metadata.get("surrounding_text", ""),
                    "text": text,
                }
            )

        return sorted(figures, key=lambda f: (str(f.get("filename", "")), int(f.get("page", 1))))


_store: ChromaStore | None = None


def get_store() -> ChromaStore:
    global _store
    if _store is None:
        _store = ChromaStore()
    return _store

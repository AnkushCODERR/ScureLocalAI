from pathlib import Path
import pytest

from backend.config import (
    EmbeddingSettings,
    LlmSettings,
    RetrievalSettings,
    ServerSettings,
    Settings,
    StorageSettings,
    VisionSettings,
)
from backend.ingestion.Visualizing import build_selftest_pdf
from backend.ingestion.pipeline import ingest_document


class MockStore:
    def __init__(self):
        self.chunks = []
        self.added = None

    def add_chunks(self, doc_id, filename, chunks, embeddings):
        self.added = (doc_id, filename, chunks, embeddings)
        self.chunks.extend(chunks)


@pytest.mark.asyncio
async def test_multimodal_ingestion_indexes_text_and_figures(tmp_path: Path, monkeypatch):
    # Build synthetic test PDF with figures
    pdf_path = tmp_path / "test_doc.pdf"
    build_selftest_pdf(pdf_path)

    store = MockStore()
    settings = Settings(
        llm=LlmSettings("qwen3:4b", "http://localhost:11434", 0.1, 128),
        embeddings=EmbeddingSettings("nomic-embed-text", "http://localhost:11434"),
        retrieval=RetrievalSettings(chunk_size=512, chunk_overlap=32, top_k=5),
        storage=StorageSettings(tmp_path / "documents", tmp_path / "chroma", "test"),
        server=ServerSettings("127.0.0.1", 8000),
        vision=VisionSettings(
            output_dir=tmp_path / "figures",
            crop_dpi=90,
            enabled=True,
        ),
    )

    async def fake_embed(texts, settings):
        return [[0.1] * 8 for _ in texts]

    async def fake_describe(image_path, caption=None, kind="unknown", page_context=None, settings=None):
        return f"AI visual breakdown of {kind} with caption {caption}"

    monkeypatch.setattr("backend.ingestion.pipeline.embed", fake_embed)
    monkeypatch.setattr("backend.ingestion.pipeline.describe_document_figure", fake_describe)

    total_chunks = await ingest_document(pdf_path, "doc-123", "test_doc.pdf", settings=settings, store=store)

    assert total_chunks > 0
    # Verify that both text chunks and figure chunks were added
    all_chunks = store.added[2]
    figure_chunks = [c for c in all_chunks if c.get("is_image") is True]
    text_chunks = [c for c in all_chunks if not c.get("is_image")]

    assert len(figure_chunks) > 0, "Expected figure chunks to be extracted"
    assert len(text_chunks) > 0, "Expected text chunks to be extracted"

    # Verify figure chunk metadata links image with page and context
    first_fig = figure_chunks[0]
    assert first_fig["page"] >= 1
    assert "image_url" in first_fig
    assert "/api/figures/doc-123/crops/" in first_fig["image_url"]
    assert "Visual Explanation" in first_fig["text"]

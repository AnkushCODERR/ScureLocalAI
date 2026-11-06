import io
from pathlib import Path
from PIL import Image
import pytest

from backend.config import VisionSettings, Settings, LlmSettings, EmbeddingSettings, RetrievalSettings, StorageSettings, ServerSettings
from backend.llm.vision import (
    describe_document_figure,
    explain_image_detail,
    check_image_relevance,
    encode_image,
    _heuristic_figure_fallback,
)


def _create_sample_image(tmp_path: Path) -> Path:
    img_path = tmp_path / "sample.png"
    img = Image.new("RGB", (100, 80), color=(73, 109, 137))
    img.save(img_path)
    return img_path


def test_encode_image_resizes_and_returns_base64(tmp_path: Path):
    img_path = _create_sample_image(tmp_path)
    b64 = encode_image(img_path, max_dimension=50)
    assert isinstance(b64, str)
    assert len(b64) > 10


def test_encode_image_handles_data_url(tmp_path: Path):
    import base64
    img_path = _create_sample_image(tmp_path)
    raw_b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")
    data_url = f"data:image/png;base64,{raw_b64}"

    encoded = encode_image(data_url, max_dimension=50)
    assert isinstance(encoded, str)
    assert len(encoded) > 10


def test_encode_image_handles_raw_base64(tmp_path: Path):
    import base64
    img_path = _create_sample_image(tmp_path)
    raw_b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")

    encoded = encode_image(raw_b64, max_dimension=50)
    assert isinstance(encoded, str)
    assert len(encoded) > 10


def test_figure_crop_endpoint_serves_image(tmp_path: Path):
    from fastapi.testclient import TestClient
    from backend.main import app
    from backend.config import get_settings

    settings = get_settings()
    doc_id = "test-doc-id"
    crops_dir = settings.vision.output_dir / doc_id / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGB", (60, 60), color=(255, 0, 0))
    crop_path = crops_dir / "p001_f01.png"
    img.save(crop_path, format="PNG")

    client = TestClient(app)
    response = client.get(f"/api/figures/{doc_id}/crops/p001_f01.png")

    assert response.status_code == 200
    assert response.headers.get("content-type") == "image/png"
    assert len(response.content) > 0


def test_heuristic_fallback_when_no_caption():
    text = _heuristic_figure_fallback("chart", caption=None, page_context="Quarterly growth was observed.")
    assert "Chart or graphical data visualization" in text
    assert "no caption provided in the document" in text.lower()
    assert "Quarterly growth" in text


@pytest.mark.asyncio
async def test_describe_document_figure_without_caption(tmp_path: Path, monkeypatch):
    img_path = _create_sample_image(tmp_path)

    # Mock call_vlm to return a detailed explanation
    async def fake_call_vlm(image_input, prompt, settings=None, timeout=60.0):
        return "This is a bar chart comparing sales over Q1 and Q2. Q2 shows a 30% increase."

    monkeypatch.setattr("backend.llm.vision.call_vlm", fake_call_vlm)

    explanation = await describe_document_figure(
        image_path=img_path,
        caption=None,  # No description in document
        kind="chart",
        page_context="Page context text",
    )

    assert "bar chart comparing sales" in explanation


@pytest.mark.asyncio
async def test_check_image_relevance_matches_documents(tmp_path: Path, monkeypatch):
    img_path = _create_sample_image(tmp_path)

    async def fake_call_vlm(image_input, prompt, settings=None, timeout=60.0):
        return "System architecture diagram illustrating client, API gateway, and microservices."

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "message": {
                    "content": (
                        "RELEVANCE: Highly Relevant\n"
                        "SCORE: 88%\n"
                        "RATIONALE: The architecture diagram closely aligns with the microservices spec in report.pdf page 3."
                    )
                }
            }

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr("backend.llm.vision.call_vlm", fake_call_vlm)
    monkeypatch.setattr("backend.llm.vision.httpx.AsyncClient", FakeAsyncClient)

    settings = Settings(
        llm=LlmSettings("qwen3:4b", "http://localhost:11434", 0.1, 128),
        embeddings=EmbeddingSettings("nomic-embed-text", "http://localhost:11434"),
        retrieval=RetrievalSettings(512, 64, 5),
        storage=StorageSettings(tmp_path / "documents", tmp_path / "chroma", "test"),
        server=ServerSettings("127.0.0.1", 8000),
    )

    chunks = [
        {"filename": "spec.pdf", "page": 3, "text": "Our microservices architecture features an API gateway.", "distance": 0.15}
    ]

    res = await check_image_relevance(img_path, "Does this match architecture?", chunks, settings)

    assert res["is_relevant"] is True
    assert res["relevance_score"] == 88
    assert res["relevance_label"] == "Highly Relevant"
    assert "report.pdf page 3" in res["analysis"]
    assert len(res["matching_chunks"]) == 1

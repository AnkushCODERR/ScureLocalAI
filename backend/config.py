from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"


@dataclass(frozen=True)
class LlmSettings:
    model: str
    base_url: str
    temperature: float
    max_tokens: int


@dataclass(frozen=True)
class EmbeddingSettings:
    model: str
    base_url: str


@dataclass(frozen=True)
class RetrievalSettings:
    chunk_size: int
    chunk_overlap: int
    top_k: int


@dataclass(frozen=True)
class StorageSettings:
    documents_dir: Path
    chroma_dir: Path
    chroma_collection: str


@dataclass(frozen=True)
class ServerSettings:
    host: str
    port: int


@dataclass(frozen=True)
class VisionSettings:
    model: str = "gemma3:4b"
    base_url: str = "http://localhost:11434"
    enabled: bool = True
    output_dir: Path = ROOT_DIR / "data" / "figures"
    crop_dpi: int = 150
    min_side_pt: int = 28
    cluster_gap_pt: int = 9
    vector_min_score: float = 0.45
    drop_recurring: bool = True
    raster_enabled: bool = True


@dataclass(frozen=True)
class Settings:
    llm: LlmSettings
    embeddings: EmbeddingSettings
    retrieval: RetrievalSettings
    storage: StorageSettings
    server: ServerSettings
    vision: VisionSettings = field(default_factory=VisionSettings)


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT_DIR / path


def _read_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("config.yaml must contain a mapping at the top level")
    return data


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    data = _read_config()
    llm = data.get("llm", {})
    embeddings = data.get("embeddings", {})
    retrieval = data.get("retrieval", {})
    storage = data.get("storage", {})
    server = data.get("server", {})
    vision = data.get("vision", {})

    return Settings(
        llm=LlmSettings(
            model=str(llm["model"]),
            base_url=str(llm["base_url"]).rstrip("/"),
            temperature=float(llm.get("temperature", 0.1)),
            max_tokens=int(llm.get("max_tokens", 2048)),
        ),
        embeddings=EmbeddingSettings(
            model=str(embeddings["model"]),
            base_url=str(embeddings["base_url"]).rstrip("/"),
        ),
        retrieval=RetrievalSettings(
            chunk_size=int(retrieval.get("chunk_size", 512)),
            chunk_overlap=int(retrieval.get("chunk_overlap", 64)),
            top_k=int(retrieval.get("top_k", 5)),
        ),
        storage=StorageSettings(
            documents_dir=_project_path(str(storage.get("documents_dir", "data/documents"))),
            chroma_dir=_project_path(str(storage.get("chroma_dir", "data/chroma_db"))),
            chroma_collection=str(storage.get("chroma_collection", "scurelocalai")),
        ),
        server=ServerSettings(
            host=str(server.get("host", "127.0.0.1")),
            port=int(server.get("port", 8000)),
        ),
        vision=VisionSettings(
            model=str(vision.get("model", "gemma3:4b")),
            base_url=str(vision.get("base_url", llm.get("base_url", "http://localhost:11434"))).rstrip("/"),
            enabled=bool(vision.get("enabled", True)),
            output_dir=_project_path(str(vision.get("output_dir", "data/figures"))),
            crop_dpi=int(vision.get("crop_dpi", 150)),
            min_side_pt=int(vision.get("min_side_pt", 28)),
            cluster_gap_pt=int(vision.get("cluster_gap_pt", 9)),
            vector_min_score=float(vision.get("vector_min_score", 0.45)),
            drop_recurring=bool(vision.get("drop_recurring", True)),
            raster_enabled=bool(vision.get("raster_enabled", True)),
        ),
    )


def ensure_storage_dirs(settings: Settings | None = None) -> None:
    resolved = settings or get_settings()
    resolved.storage.documents_dir.mkdir(parents=True, exist_ok=True)
    resolved.storage.chroma_dir.mkdir(parents=True, exist_ok=True)
    resolved.vision.output_dir.mkdir(parents=True, exist_ok=True)

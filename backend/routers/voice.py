from __future__ import annotations

import re
from fastapi import APIRouter
from pydantic import BaseModel


router = APIRouter(prefix="/voice", tags=["voice"])


class CleanSpeechRequest(BaseModel):
    text: str


class CleanSpeechResponse(BaseModel):
    speech_text: str


@router.get("/status")
async def voice_status() -> dict[str, object]:
    return {
        "status": "ready",
        "engine": "Web Speech API (client-side offline TTS & STT)",
        "features": [
            "SpeechSynthesis (Text-to-Speech) for answering questions aloud",
            "SpeechRecognition (Speech-to-Text) for voice question input",
            "Automatic sentence chunking for uninterrupted narration",
            "Audio wave visualizer and speech playback controls",
        ],
    }


@router.post("/clean-for-speech", response_model=CleanSpeechResponse)
async def clean_for_speech(request: CleanSpeechRequest) -> CleanSpeechResponse:
    raw = request.text

    # Strip code blocks
    cleaned = re.sub(r"```[\s\S]*?```", " Code snippet omitted for voice playback. ", raw)
    # Strip inline code
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    # Strip markdown headers (### Header -> Header)
    cleaned = re.sub(r"^#+\s*", "", cleaned, flags=re.MULTILINE)
    # Strip markdown bold / italic
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
    cleaned = re.sub(r"__([^_]+)__", r"\1", cleaned)
    cleaned = re.sub(r"_([^_]+)_", r"\1", cleaned)
    # Strip URLs
    cleaned = re.sub(r"https?://\S+", "", cleaned)
    # Strip bullet stars/dashes
    cleaned = re.sub(r"^\s*[-*•]\s*", "", cleaned, flags=re.MULTILINE)
    # Clean Sources trailing section
    cleaned = re.sub(r"Sources:[\s\S]*$", "", cleaned, flags=re.IGNORECASE)
    # Collapse multiple whitespaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return CleanSpeechResponse(speech_text=cleaned)

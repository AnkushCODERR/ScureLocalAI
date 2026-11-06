from __future__ import annotations

import json
import logging
import re
from typing import Any, AsyncIterator

import httpx

from backend.config import LlmSettings, get_settings


logger = logging.getLogger(__name__)

NOT_FOUND_MESSAGE = "I could not find relevant information in the knowledge base."

RAG_SYSTEM_PROMPT = """You are a multimodal research assistant. Answer ONLY using the provided document excerpts and visual figure descriptions.
You can understand and explain charts, diagrams, flowcharts, photographs, and tables described in the excerpts, and link them to the surrounding text.
If the excerpts do not contain enough information to answer the question, respond with:
"I could not find relevant information in the knowledge base."
Always end your answer with a "Sources:" section listing the filenames you used."""


def _format_context(context_chunks: list[dict[str, Any]]) -> str:
    excerpts: list[str] = []
    for index, chunk in enumerate(context_chunks, start=1):
        filename = chunk.get("filename", "unknown")
        page = chunk.get("page", "unknown")
        text = chunk.get("text", "")
        if chunk.get("is_image"):
            kind = chunk.get("kind", "Visual")
            caption = chunk.get("caption", "")
            label = f" [Figure ({kind}): {caption}]" if caption else f" [Figure ({kind})]"
            excerpts.append(f"[{index}] Source: {filename}, page {page}{label}\n{text}")
        else:
            excerpts.append(f"[{index}] Source: {filename}, page {page}\n{text}")
    return "\n\n".join(excerpts)


def _source_names(context_chunks: list[dict[str, Any]]) -> list[str]:
    names = []
    for chunk in context_chunks:
        filename = str(chunk.get("filename", "")).strip()
        if filename and filename not in names:
            names.append(filename)
    return names


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def ensure_sources(answer: str, context_chunks: list[dict[str, Any]]) -> str:
    sources = _source_names(context_chunks)
    if not sources:
        return answer

    if "Sources:" in answer:
        return answer.strip()

    source_lines = "\n".join(f"- {source}" for source in sources)
    return f"{answer.strip()}\n\nSources:\n{source_lines}"


async def answer_query(
    question: str,
    context_chunks: list[dict[str, Any]],
    settings: LlmSettings | None = None,
) -> str:
    if not context_chunks:
        return NOT_FOUND_MESSAGE

    resolved = settings or get_settings().llm
    context = _format_context(context_chunks)
    messages = [
        {"role": "system", "content": f"{RAG_SYSTEM_PROMPT}\n\nDocument excerpts:\n{context}"},
        {"role": "user", "content": question},
    ]
    payload = {
        "model": resolved.model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": resolved.temperature,
            "num_predict": resolved.max_tokens,
        },
    }

    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(f"{resolved.base_url}/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()

    try:
        answer = str(data["message"]["content"])
    except KeyError as exc:
        logger.error("Unexpected Ollama chat response: %s", data)
        raise RuntimeError("Ollama returned an unexpected chat response") from exc

    return ensure_sources(_strip_thinking(answer), context_chunks)


class StreamThinkingFilter:
    """Filter out <think>...</think> thinking tokens from a streaming LLM response."""

    def __init__(self) -> None:
        self.in_think = False
        self.buffer = ""

    def process(self, token: str) -> str:
        self.buffer += token
        output = ""
        while self.buffer:
            if not self.in_think:
                if "<think>" in self.buffer:
                    before, _, after = self.buffer.partition("<think>")
                    output += before
                    self.in_think = True
                    self.buffer = after
                elif any("<think>".startswith(self.buffer[i:]) for i in range(len(self.buffer))):
                    match_pos = min(i for i in range(len(self.buffer)) if "<think>".startswith(self.buffer[i:]))
                    output += self.buffer[:match_pos]
                    self.buffer = self.buffer[match_pos:]
                    break
                else:
                    output += self.buffer
                    self.buffer = ""
            else:
                if "</think>" in self.buffer:
                    _, _, after = self.buffer.partition("</think>")
                    self.in_think = False
                    self.buffer = after
                else:
                    for i in range(len(self.buffer)):
                        if "</think>".startswith(self.buffer[i:]):
                            self.buffer = self.buffer[i:]
                            break
                    else:
                        self.buffer = ""
                    break
        return output

    def flush(self) -> str:
        if not self.in_think and self.buffer:
            out = self.buffer
            self.buffer = ""
            return out
        return ""


async def answer_query_stream(
    question: str,
    context_chunks: list[dict[str, Any]],
    settings: LlmSettings | None = None,
) -> AsyncIterator[str]:
    """Stream answer tokens directly from Ollama as they are generated."""
    if not context_chunks:
        yield NOT_FOUND_MESSAGE
        return

    resolved = settings or get_settings().llm
    context = _format_context(context_chunks)
    messages = [
        {"role": "system", "content": f"{RAG_SYSTEM_PROMPT}\n\nDocument excerpts:\n{context}"},
        {"role": "user", "content": question},
    ]
    payload = {
        "model": resolved.model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": resolved.temperature,
            "num_predict": resolved.max_tokens,
        },
    }

    thinking_filter = StreamThinkingFilter()
    async with httpx.AsyncClient(timeout=180.0) as client:
        async with client.stream("POST", f"{resolved.base_url}/api/chat", json=payload) as response:
            response.raise_for_status()
            async for raw_line in response.aiter_lines():
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue

                content = data.get("message", {}).get("content", "")
                if not content:
                    continue

                filtered = thinking_filter.process(content)
                if filtered:
                    yield filtered

            final_token = thinking_filter.flush()
            if final_token:
                yield final_token

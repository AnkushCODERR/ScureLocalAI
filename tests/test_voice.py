from backend.routers.voice import clean_for_speech, CleanSpeechRequest
import pytest


@pytest.mark.asyncio
async def test_clean_for_speech_strips_markdown_and_citations():
    raw_text = (
        "### Summary of Findings\n"
        "The **revenue** increased by *15%* as shown in `Table 1`.\n"
        "```python\nprint('code block')\n```\n"
        "- Point 1\n"
        "- Point 2\n"
        "Visit https://example.com for info.\n\n"
        "Sources:\n- report.pdf"
    )
    req = CleanSpeechRequest(text=raw_text)
    res = await clean_for_speech(req)

    assert "###" not in res.speech_text
    assert "**" not in res.speech_text
    assert "```" not in res.speech_text
    assert "https://" not in res.speech_text
    assert "Sources:" not in res.speech_text
    assert "revenue increased by 15% as shown in Table 1." in res.speech_text

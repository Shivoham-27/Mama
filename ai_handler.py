"""
ai_handler.py – Unified interface for OpenAI, Gemini, Anthropic, and OpenRouter.
Each provider receives the conversation history list and returns a text reply.

History item schema (internal):
  { "role": "user" | "assistant", "content": str | list[part] }

A "part" for multimodal messages:
  { "type": "text",       "text": "..." }
  { "type": "image_b64",  "data": "<base64>", "mime": "image/jpeg" }
  { "type": "pdf_text",   "text": "...extracted text..." }
"""

import base64
import httpx
from config import AI_PROVIDER, AI_API_KEY, get_model

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# ──────────────────────────────────────────────
# OpenAI
# ──────────────────────────────────────────────
def _build_openai_messages(history: list[dict]) -> list[dict]:
    messages = []
    for item in history:
        role = item["role"]
        content = item["content"]

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
        else:
            parts = []
            for part in content:
                if part["type"] == "text":
                    parts.append({"type": "text", "text": part["text"]})
                elif part["type"] == "image_b64":
                    parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{part['mime']};base64,{part['data']}"
                        },
                    })
                elif part["type"] == "pdf_text":
                    parts.append({
                        "type": "text",
                        "text": f"[PDF Content]\n{part['text']}"
                    })
            messages.append({"role": role, "content": parts})
    return messages


def _ask_openai(history: list[dict]) -> str:
    import openai
    client = openai.OpenAI(api_key=AI_API_KEY)
    response = client.chat.completions.create(
        model=get_model(),
        messages=_build_openai_messages(history),
    )
    return response.choices[0].message.content.strip()


# ──────────────────────────────────────────────
# OpenRouter  (OpenAI-compatible, any model)
# ──────────────────────────────────────────────
def _ask_openrouter(history: list[dict]) -> str:
    import openai
    client = openai.OpenAI(
        api_key=AI_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )
    response = client.chat.completions.create(
        model=get_model(),
        messages=_build_openai_messages(history),
    )
    return response.choices[0].message.content.strip()


# ──────────────────────────────────────────────
# Google Gemini
# ──────────────────────────────────────────────
def _build_gemini_contents(history: list[dict]) -> list[dict]:
    contents = []
    for item in history:
        role = "user" if item["role"] == "user" else "model"
        content = item["content"]

        if isinstance(content, str):
            parts = [{"text": content}]
        else:
            parts = []
            for part in content:
                if part["type"] == "text":
                    parts.append({"text": part["text"]})
                elif part["type"] == "image_b64":
                    parts.append({
                        "inlineData": {
                            "mimeType": part["mime"],
                            "data": part["data"],
                        }
                    })
                elif part["type"] == "pdf_text":
                    parts.append({"text": f"[PDF Content]\n{part['text']}"})
        contents.append({"role": role, "parts": parts})
    return contents


def _ask_gemini(history: list[dict]) -> str:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{get_model()}:generateContent?key={AI_API_KEY}"
    )
    payload = {"contents": _build_gemini_contents(history)}
    resp = httpx.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


# ──────────────────────────────────────────────
# Anthropic Claude
# ──────────────────────────────────────────────
def _build_anthropic_messages(history: list[dict]) -> list[dict]:
    messages = []
    for item in history:
        role = item["role"]
        content = item["content"]

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
        else:
            parts = []
            for part in content:
                if part["type"] == "text":
                    parts.append({"type": "text", "text": part["text"]})
                elif part["type"] == "image_b64":
                    parts.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": part["mime"],
                            "data": part["data"],
                        },
                    })
                elif part["type"] == "pdf_text":
                    parts.append({
                        "type": "text",
                        "text": f"[PDF Content]\n{part['text']}"
                    })
            messages.append({"role": role, "content": parts})
    return messages


def _ask_anthropic(history: list[dict]) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=AI_API_KEY)
    response = client.messages.create(
        model=get_model(),
        max_tokens=4096,
        messages=_build_anthropic_messages(history),
    )
    return response.content[0].text.strip()


# ──────────────────────────────────────────────
# Public interface
# ──────────────────────────────────────────────
def ask_ai(history: list[dict]) -> str:
    """Send conversation history to the configured AI provider and return reply."""
    if AI_PROVIDER == "openrouter":
        return _ask_openrouter(history)
    elif AI_PROVIDER == "openai":
        return _ask_openai(history)
    elif AI_PROVIDER == "gemini":
        return _ask_gemini(history)
    elif AI_PROVIDER == "anthropic":
        return _ask_anthropic(history)
    else:
        raise ValueError(f"Unknown AI_PROVIDER: '{AI_PROVIDER}'. Use: openrouter | openai | gemini | anthropic")


def encode_image_bytes(data: bytes, mime: str = "image/jpeg") -> dict:
    """Return a multimodal image part from raw bytes."""
    return {
        "type": "image_b64",
        "data": base64.b64encode(data).decode(),
        "mime": mime,
    }


def make_text_part(text: str) -> dict:
    return {"type": "text", "text": text}


def make_pdf_part(text: str) -> dict:
    return {"type": "pdf_text", "text": text}

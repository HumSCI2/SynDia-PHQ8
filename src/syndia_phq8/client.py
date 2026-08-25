"""Minimal client for Ollama and other OpenAI-compatible chat endpoints."""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from .config import EndpointConfig


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError("Model response did not contain a JSON object")
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Model response JSON must be an object")
    return value


def chat(
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.0,
    json_response: bool = False,
    config: EndpointConfig | None = None,
) -> str | dict[str, Any]:
    endpoint = config or EndpointConfig.from_env()
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    if json_response:
        payload["response_format"] = {"type": "json_object"}

    with httpx.Client(timeout=endpoint.timeout_seconds) as client:
        response = client.post(
            f"{endpoint.base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {endpoint.api_key}"},
            json=payload,
        )
        response.raise_for_status()
        body = response.json()
    content = body["choices"][0]["message"]["content"]
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Model returned an empty response")
    return _extract_json(content) if json_response else content

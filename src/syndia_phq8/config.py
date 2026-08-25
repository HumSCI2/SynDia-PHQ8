"""Shared configuration for OpenAI-compatible local model endpoints."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class EndpointConfig:
    """Connection settings read from environment variables."""

    base_url: str
    api_key: str
    timeout_seconds: float

    @classmethod
    def from_env(cls) -> "EndpointConfig":
        return cls(
            base_url=os.getenv("SYNDIA_BASE_URL", "http://localhost:11434/v1"),
            api_key=os.getenv("SYNDIA_API_KEY", "ollama"),
            timeout_seconds=float(os.getenv("SYNDIA_TIMEOUT_SECONDS", "300")),
        )

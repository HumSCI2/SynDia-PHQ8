"""Data contracts shared by generation, assessment, and fine-tuning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

PHQ8_ITEMS = (
    "anhedonia",
    "depressed_mood",
    "sleep",
    "fatigue",
    "appetite",
    "self_worth",
    "concentration",
    "psychomotor",
)


@dataclass(frozen=True)
class Sample:
    sample_id: str
    source: str
    profile: dict[str, Any]
    phq8_items: tuple[int, ...]
    conversations: tuple[tuple[dict[str, str], ...], ...]
    provenance: str

    @property
    def phq8_total(self) -> int:
        return sum(self.phq8_items)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "Sample":
        items = tuple(int(value) for value in row["phq8_items"])
        if len(items) != 8 or any(value not in range(4) for value in items):
            raise ValueError("phq8_items must contain eight integers in [0, 3]")
        conversations = tuple(tuple(conversation) for conversation in row["conversations"])
        if not conversations or any(
            not conversation
            or any(turn.get("role") not in {"therapist", "patient"} for turn in conversation)
            for conversation in conversations
        ):
            raise ValueError("conversations must contain therapist/patient turns")
        return cls(
            sample_id=str(row["sample_id"]),
            source=str(row["source"]),
            profile=dict(row["profile"]),
            phq8_items=items,
            conversations=conversations,
            provenance=str(row["provenance"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source": self.source,
            "profile": self.profile,
            "phq8_items": list(self.phq8_items),
            "phq8_total": self.phq8_total,
            "conversations": [list(conversation) for conversation in self.conversations],
            "provenance": self.provenance,
        }


def read_jsonl(path: str | Path) -> list[Sample]:
    samples: list[Sample] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                samples.append(Sample.from_dict(json.loads(line)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid record at {path}:{line_number}: {exc}") from exc
    return samples


def write_jsonl(samples: Iterable[Sample], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_dict(), ensure_ascii=True) + "\n")

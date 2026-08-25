"""Run structured PHQ-8 label reconstruction on shared dialogue inputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from syndia_phq8.client import chat
from syndia_phq8.prompts import ASSESSMENT_SYSTEM
from syndia_phq8.schema import PHQ8_ITEMS, Sample, read_jsonl


def severity(total: int) -> str:
    if total < 5:
        return "minimal"
    if total < 10:
        return "mild"
    if total < 15:
        return "moderate"
    if total < 20:
        return "moderately severe"
    return "severe"


def assess_sample(sample: Sample, model: str, conversation_index: int) -> dict[str, object]:
    if conversation_index >= len(sample.conversations):
        raise IndexError(f"{sample.sample_id} has no conversation {conversation_index}")
    request = {
        "phq8_item_order": list(PHQ8_ITEMS),
        "conversation": sample.conversations[conversation_index],
    }
    response = chat(
        model,
        ASSESSMENT_SYSTEM,
        json.dumps(request, ensure_ascii=True),
        temperature=0.0,
        json_response=True,
    )
    items = [int(value) for value in response.get("items", [])]
    if len(items) != 8 or any(value not in range(4) for value in items):
        raise ValueError("Assessment must contain eight item scores in [0, 3]")
    total = sum(items)
    return {
        "sample_id": sample.sample_id,
        "source": sample.source,
        "model": model,
        "conversation_index": conversation_index,
        "reference_total": sample.phq8_total,
        "predicted_total": total,
        "predicted_depressed": total >= 10,
        "predicted_severity": severity(total),
        **{f"predicted_{name}": value for name, value in zip(PHQ8_ITEMS, items)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--conversation-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    samples = read_jsonl(args.input)
    if args.limit:
        samples = samples[: args.limit]
    rows = [
        assess_sample(sample, model, args.conversation_index)
        for sample in samples
        for model in args.model
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} predictions to {output}")


if __name__ == "__main__":
    main()

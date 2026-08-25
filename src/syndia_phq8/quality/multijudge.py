"""Evaluate synthetic dialogue sets with a fixed LLM judge panel."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
from pathlib import Path

from syndia_phq8.client import chat
from syndia_phq8.prompts import QUALITY_SYSTEM
from syndia_phq8.schema import Sample, read_jsonl

DIMENSIONS = (
    "persona_fidelity",
    "symptom_fidelity",
    "therapeutic_realism",
    "linguistic_naturalness",
    "diversity",
    "constraint_adherence",
    "overall_quality",
)


def _words(sample: Sample) -> list[set[str]]:
    return [
        set(re.findall(r"[a-z']+", " ".join(turn["text"] for turn in conversation).lower()))
        for conversation in sample.conversations
    ]


def lexical_diagnostics(sample: Sample) -> dict[str, float]:
    word_sets = _words(sample)
    similarities = [
        len(left & right) / len(left | right) if left | right else 0.0
        for left, right in itertools.combinations(word_sets, 2)
    ]
    all_tokens = [
        token
        for conversation in sample.conversations
        for turn in conversation
        for token in re.findall(r"[a-z']+", turn["text"].lower())
    ]
    return {
        "conversation_count": len(sample.conversations),
        "mean_jaccard_similarity": sum(similarities) / len(similarities) if similarities else 0.0,
        "distinct_unigram_ratio": len(set(all_tokens)) / len(all_tokens) if all_tokens else 0.0,
    }


def judge_sample(sample: Sample, judge_model: str) -> dict[str, object]:
    diagnostics = lexical_diagnostics(sample)
    request = {
        "profile": sample.profile,
        "target_symptom_pattern": list(sample.phq8_items),
        "conversations": sample.conversations,
        "lexical_diagnostics": diagnostics,
        "dimensions": list(DIMENSIONS),
        "required_format": {
            dimension: {"score": "integer 1-5", "evidence": "brief string"}
            for dimension in DIMENSIONS
        },
    }
    response = chat(
        judge_model,
        QUALITY_SYSTEM,
        json.dumps(request, ensure_ascii=True),
        temperature=0.0,
        json_response=True,
    )
    row: dict[str, object] = {
        "sample_id": sample.sample_id,
        "source": sample.source,
        "judge_model": judge_model,
        **diagnostics,
    }
    for dimension in DIMENSIONS:
        rating = response.get(dimension)
        if not isinstance(rating, dict):
            raise ValueError(f"Judge response missing {dimension}")
        score = int(rating.get("score", 0))
        if score not in range(1, 6):
            raise ValueError(f"Invalid {dimension} score: {score}")
        row[dimension] = score
        row[f"{dimension}_evidence"] = str(rating.get("evidence", "")).strip()
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--judge-model", action="append", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    samples = read_jsonl(args.input)
    if args.limit:
        samples = samples[: args.limit]
    rows = [judge_sample(sample, judge) for sample in samples for judge in args.judge_model]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} judge rows to {output}")


if __name__ == "__main__":
    main()

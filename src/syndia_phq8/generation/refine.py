"""Normalize generated dialogue sets and reject prohibited leakage."""

from __future__ import annotations

import argparse
import re

from syndia_phq8.schema import Sample, read_jsonl, write_jsonl

PROHIBITED = (
    re.compile(r"\bPHQ-?8\b", re.IGNORECASE),
    re.compile(r"\b(?:score|total)\s*(?:is|=|:)\s*\d+\b", re.IGNORECASE),
    re.compile(r"\bdiagnos(?:is|ed)\b", re.IGNORECASE),
)


def refine_sample(sample: Sample) -> Sample:
    conversations = []
    for conversation in sample.conversations:
        turns = []
        previous_role = None
        for turn in conversation:
            text = " ".join(str(turn["text"]).split())
            if any(pattern.search(text) for pattern in PROHIBITED):
                raise ValueError(f"{sample.sample_id}: prohibited label or diagnosis leakage")
            role = str(turn["role"]).lower()
            if role == previous_role:
                turns[-1]["text"] = f"{turns[-1]['text']} {text}"
            else:
                turns.append({"role": role, "text": text})
            previous_role = role
        conversations.append(tuple(turns))
    return Sample(
        sample_id=sample.sample_id,
        source=sample.source,
        profile=sample.profile,
        phq8_items=sample.phq8_items,
        conversations=tuple(conversations),
        provenance=sample.provenance + "; deterministic public refinement applied",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    refined = [refine_sample(sample) for sample in read_jsonl(args.input)]
    write_jsonl(refined, args.output)
    print(f"Wrote {len(refined)} refined records to {args.output}")


if __name__ == "__main__":
    main()

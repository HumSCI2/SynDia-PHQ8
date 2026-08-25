"""Generate profile-grounded synthetic dialogue sets."""

from __future__ import annotations

import argparse
import json

from syndia_phq8.client import chat
from syndia_phq8.prompts import GENERATION_SYSTEM
from syndia_phq8.schema import Sample, read_jsonl, write_jsonl


def _validate_conversations(value: object, expected_count: int) -> tuple[tuple[dict[str, str], ...], ...]:
    if not isinstance(value, list) or len(value) != expected_count:
        raise ValueError(f"Expected exactly {expected_count} conversations")
    validated = []
    for conversation in value:
        if not isinstance(conversation, list) or not conversation:
            raise ValueError("Each conversation must be a non-empty list")
        turns = []
        for turn in conversation:
            if not isinstance(turn, dict) or turn.get("role") not in {"therapist", "patient"}:
                raise ValueError("Each turn requires a therapist/patient role")
            text = str(turn.get("text", "")).strip()
            if not text:
                raise ValueError("Conversation turns cannot be empty")
            turns.append({"role": turn["role"], "text": text})
        validated.append(tuple(turns))
    return tuple(validated)


def generate_sample(sample: Sample, model: str, conversation_count: int) -> Sample:
    prompt = json.dumps(
        {
            "synthetic_profile": sample.profile,
            "target_symptom_pattern": list(sample.phq8_items),
            "conversation_count": conversation_count,
            "instructions": "Use approximately ten alternating turns per conversation.",
        },
        ensure_ascii=True,
    )
    response = chat(model, GENERATION_SYSTEM, prompt, temperature=0.7, json_response=True)
    conversations = _validate_conversations(response.get("conversations"), conversation_count)
    return Sample(
        sample_id=sample.sample_id,
        source=sample.source,
        profile=sample.profile,
        phq8_items=sample.phq8_items,
        conversations=conversations,
        provenance=f"Generated with {model} from a fully synthetic public profile",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input JSONL containing synthetic profiles")
    parser.add_argument("--output", required=True, help="Output JSONL for generated dialogue sets")
    parser.add_argument("--model", required=True, help="Generator model name")
    parser.add_argument("--conversation-count", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    samples = read_jsonl(args.input)
    if args.limit:
        samples = samples[: args.limit]
    generated = [generate_sample(sample, args.model, args.conversation_count) for sample in samples]
    write_jsonl(generated, args.output)
    print(f"Wrote {len(generated)} generated records to {args.output}")


if __name__ == "__main__":
    main()

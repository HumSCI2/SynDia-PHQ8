"""Prepare instruction records and optionally fine-tune a causal LM with LoRA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from syndia_phq8.schema import PHQ8_ITEMS, read_jsonl


def format_records(input_path: str, conversation_index: int) -> list[dict[str, str]]:
    records = []
    for sample in read_jsonl(input_path):
        if conversation_index >= len(sample.conversations):
            raise IndexError(f"{sample.sample_id} has no conversation {conversation_index}")
        dialogue = "\n".join(
            f"{turn['role'].title()}: {turn['text']}"
            for turn in sample.conversations[conversation_index]
        )
        target = {
            "items": dict(zip(PHQ8_ITEMS, sample.phq8_items)),
            "total": sample.phq8_total,
            "depressed": sample.phq8_total >= 10,
        }
        records.append(
            {
                "sample_id": sample.sample_id,
                "text": (
                    "Reconstruct the PHQ-8 labels from this dialogue. Return JSON only.\n\n"
                    f"{dialogue}\n\nAssistant: {json.dumps(target, ensure_ascii=True)}"
                ),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--conversation-index", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    args = parser.parse_args()

    records = format_records(args.input, args.conversation_index)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = output_dir / "prepared_records.jsonl"
    with prepared_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    print(f"Prepared {len(records)} records at {prepared_path}")
    if args.prepare_only:
        return

    try:
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import SFTTrainer
    except ImportError as exc:
        raise SystemExit("Install training dependencies with: pip install -e '.[train]'") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")
    lora = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        max_grad_norm=0.5,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(records),
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(str(output_dir / "adapter"))


if __name__ == "__main__":
    main()

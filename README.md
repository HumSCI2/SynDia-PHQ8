# SynDia-PHQ8

Reference code for **Generating and Assessing Synthetic Clinical Dialogues for PHQ-8 Reconstruction: A Multi-Model Benchmark and Fine-Tuning Study**.

SynDia-PHQ8 provides a compact, publication-facing implementation of four stages:

1. profile-grounded synthetic dialogue generation and refinement;
2. multi-judge conversation-quality evaluation;
3. controlled shared-input PHQ-8 label reconstruction; and
4. LoRA data preparation and fine-tuning.

> **Research use only.** This repository does not provide diagnosis, crisis assessment, or treatment. Model outputs require independent validation and appropriate human oversight.

## Repository layout

```text
SynDia-PHQ8/
├── configs/                 Paper model panels and defaults
├── data/samples/            10 DAIC-WOZ-labeled + 10 E-DAIC-labeled synthetic demos
├── docs/                    Data and reproducibility documentation
├── scripts/                 End-to-end sample commands
├── src/syndia_phq8/
│   ├── generation/          Dialogue generation and deterministic refinement
│   ├── quality/             Seven-dimension multi-judge evaluation
│   ├── assessment/          Structured PHQ-8 reconstruction
│   └── finetune/            LoRA preparation and training entry point
└── tests/                   Offline contract and metric tests
```

## Data availability and privacy

DAIC-WOZ and E-DAIC source records are governed by their respective access terms and are **not redistributed here**. Existing local profiles and generated artifacts can retain participant identifiers or sensitive biographical content.

The included 20 records are newly written, fully synthetic demonstrations. They are balanced across five PHQ-8 severity bands and labeled by source only to demonstrate the two-source pipeline. They are not selected, copied, or paraphrased participant records and must not be used to reproduce the paper's numerical results. See [docs/DATA_CARD.md](docs/DATA_CARD.md).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
```

Training dependencies are optional:

```bash
pip install -e '.[train,test]'
```

The model-facing commands use an OpenAI-compatible endpoint such as Ollama:

```bash
export SYNDIA_BASE_URL=http://localhost:11434/v1
export SYNDIA_API_KEY=ollama
export SYNDIA_TIMEOUT_SECONDS=300
```

## Quick start

Validate and refine the included examples without a model call:

```bash
syndia-refine \
  --input data/samples/daic_woz_synthetic.jsonl \
  --output outputs/daic_refined.jsonl
```

Generate five new dialogues per synthetic profile:

```bash
syndia-generate \
  --input data/samples/daic_woz_synthetic.jsonl \
  --output outputs/daic_generated.jsonl \
  --model gpt-oss:20b
```

Run the three-judge quality panel:

```bash
syndia-quality \
  --input outputs/daic_refined.jsonl \
  --output outputs/quality_panel.csv \
  --judge-model gpt-oss:120b \
  --judge-model gpt-oss:20b \
  --judge-model qwen3.5:9b

syndia-quality-aggregate \
  --input outputs/quality_panel.csv \
  --output outputs/quality_summary.csv
```

Run matched-input PHQ-8 reconstruction with one or more assessors:

```bash
syndia-assess \
  --input outputs/daic_refined.jsonl \
  --output outputs/phq8_predictions.csv \
  --model gpt-oss:120b \
  --model gpt-oss:20b
```

Inspect the exact instruction records before GPU training:

```bash
syndia-train \
  --input data/samples/edaic_synthetic.jsonl \
  --output-dir outputs/training_preview \
  --prepare-only
```

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the mapping between manuscript stages and commands.

## Tests

```bash
pytest -q
```

## License

Code is released under the MIT License. This license does not grant rights to DAIC-WOZ, E-DAIC, model weights, or other third-party assets.

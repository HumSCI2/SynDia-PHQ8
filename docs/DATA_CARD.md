# Data card for public demonstrations

## Contents

| File | Records | Source label | Purpose |
|---|---:|---|---|
| `data/samples/daic_woz_synthetic.jsonl` | 10 | DAIC-WOZ | Demonstrate the DAIC-WOZ branch |
| `data/samples/edaic_synthetic.jsonl` | 10 | E-DAIC | Demonstrate the E-DAIC branch |

Each record contains a synthetic profile, eight item values in PHQ-8 order, five abbreviated therapist-patient dialogues, and provenance text.

## Construction

These records were manually constructed for software demonstration. Each source-labeled set contains two examples in each band: minimal, mild, moderate, moderately severe, and severe. The source label demonstrates routing only; it does not indicate that text was sampled from that dataset.

No record was copied, selected, or paraphrased from a DAIC-WOZ or E-DAIC participant. The examples contain no real participant IDs, names, locations, employers, dates, or original transcript excerpts.

## Intended use

- testing JSONL parsing and validation;
- demonstrating generation, quality evaluation, and assessment commands;
- inspecting LoRA instruction formatting;
- writing offline unit tests.

## Out-of-scope use

- reproducing paper metrics;
- estimating clinical or demographic performance;
- training a clinically useful model;
- diagnosis, treatment, triage, or crisis assessment.

## Restricted source data

Researchers seeking to reproduce the paper must obtain DAIC-WOZ and E-DAIC through their official access processes and comply with their licenses, consent conditions, and institutional governance requirements. Do not commit source transcripts, participant profiles, or participant-linked generated text to this repository.

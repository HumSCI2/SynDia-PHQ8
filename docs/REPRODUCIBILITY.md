# Reproducibility guide

## Paper-to-code map

| Manuscript stage | Public command | Main module |
|---|---|---|
| Profile-grounded generation | `syndia-generate` | `generation/generate.py` |
| Validation and refinement | `syndia-refine` | `generation/refine.py` |
| Multi-judge quality panel | `syndia-quality` | `quality/multijudge.py` |
| Matched panel aggregation | `syndia-quality-aggregate` | `quality/aggregate.py` |
| Shared-input PHQ-8 assessment | `syndia-assess` | `assessment/evaluate.py` |
| LoRA preparation and training | `syndia-train` | `finetune/train_lora.py` |

## Experimental invariants

1. Keep generator identity fixed when comparing PHQ-8 assessors.
2. Use temperature 0 for judging and PHQ-8 reconstruction.
3. Compare generators on a complete-case source-specific intersection.
4. Keep training, development, and test participants disjoint.
5. Select checkpoints on development data only; evaluate the selected checkpoint once on test data.
6. Treat PHQ-8 output as label reconstruction rather than clinical diagnosis.

## Paper configuration

The paper evaluates eight generators/assessors and uses three fixed quality judges. Machine-readable names are in `configs/paper_models.json`.

The public LoRA entry point exposes the principal paper defaults: rank 32, alpha 64, zero dropout, five epochs, learning rate 1e-5, batch size 1 with 16 accumulation steps, cosine scheduling, 10% warmup, weight decay 0.01, and gradient clipping at 0.5. The original study additionally used model-specific MXFP4 loading and expert-parameter adaptation; these require compatible gpt-oss training software and hardware and are not silently emulated by this portable implementation.

## Reproducing numerical results

The included demonstrations are not the study cohort. Numerical reproduction requires authorized source data, the study's participant-disjoint splits, the listed model versions, and the same inference endpoint behavior. Record model revisions, endpoint versions, prompts, random seeds, valid-output coverage, and hardware with every run.

from pathlib import Path

from syndia_phq8.assessment.evaluate import severity
from syndia_phq8.finetune.train_lora import format_records
from syndia_phq8.generation.refine import refine_sample
from syndia_phq8.quality.multijudge import lexical_diagnostics
from syndia_phq8.schema import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_FILES = (
    ROOT / "data/samples/daic_woz_synthetic.jsonl",
    ROOT / "data/samples/edaic_synthetic.jsonl",
)


def test_public_samples_are_balanced_and_synthetic():
    for path in SAMPLE_FILES:
        samples = read_jsonl(path)
        assert len(samples) == 10
        assert all(len(sample.conversations) == 5 for sample in samples)
        assert all(sample.profile["synthetic"] is True for sample in samples)
        bands = [sample.profile["severity_band"] for sample in samples]
        assert all(bands.count(band) == 2 for band in set(bands))


def test_refinement_and_diagnostics():
    sample = read_jsonl(SAMPLE_FILES[0])[0]
    refined = refine_sample(sample)
    diagnostics = lexical_diagnostics(refined)
    assert diagnostics["conversation_count"] == 5
    assert 0 <= diagnostics["mean_jaccard_similarity"] <= 1
    assert 0 < diagnostics["distinct_unigram_ratio"] <= 1


def test_training_format_and_severity_boundaries():
    records = format_records(str(SAMPLE_FILES[1]), conversation_index=0)
    assert len(records) == 10
    assert records[0]["text"].endswith("}")
    assert [severity(value) for value in (0, 5, 10, 15, 20)] == [
        "minimal",
        "mild",
        "moderate",
        "moderately severe",
        "severe",
    ]

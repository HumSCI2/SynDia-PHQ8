"""Aggregate judge scores on matched source-specific sample intersections."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .multijudge import DIMENSIONS


def aggregate(input_csv: str | Path, output_csv: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(input_csv)
    required = {"sample_id", "source", "judge_model", *DIMENSIONS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    judge_count = frame["judge_model"].nunique()
    complete = frame.groupby(["source", "sample_id"])["judge_model"].nunique()
    keep = complete[complete == judge_count].rename("judge_count").reset_index()
    matched = frame.merge(keep, on=["source", "sample_id"], how="inner")
    participant = matched.groupby(["source", "sample_id"], as_index=False)[list(DIMENSIONS)].mean()
    summary = participant.groupby("source")[list(DIMENSIONS)].agg(["mean", "std"])
    summary.columns = [f"{dimension}_{stat}" for dimension, stat in summary.columns]
    summary = summary.reset_index()
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = aggregate(args.input, args.output)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

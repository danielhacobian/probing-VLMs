#!/usr/bin/env python3
"""Collect confirmatory notebook exports into the repository results folder."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CSV_EXPORTS = (
    "validation_selection_scores.csv",
    "headline_selected_test_metrics.csv",
    "headline_straightening_deltas.csv",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty confirmatory export: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect(inputs: dict[str, Path], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    protocols = {}
    for filename in CSV_EXPORTS:
        combined = []
        for environment, directory in inputs.items():
            source = directory / filename
            if not source.is_file():
                raise FileNotFoundError(source)
            rows = read_rows(source)
            if not rows:
                raise ValueError(f"{source} contains no result rows")
            combined.extend({"environment": environment, **row} for row in rows)
        write_rows(output / filename, combined)

    for environment, directory in inputs.items():
        source = directory / "headline_protocol.json"
        if not source.is_file():
            raise FileNotFoundError(source)
        protocols[environment] = json.loads(source.read_text())

    payload = {
        "protocol_version": "trajectory_grouped_60_20_20_v1",
        "selection_split": "validation_trajectories",
        "evaluation_split": "locked_test_trajectories",
        "bootstrap_unit": "complete_trajectory_window",
        "bootstrap_repeats": 1000,
        "interval": "95_percentile",
        "environments": protocols,
    }
    (output / "headline_protocol.json").write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--umaze", required=True, type=Path)
    parser.add_argument("--wall", required=True, type=Path)
    parser.add_argument("--pusht", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    collect(
        {"umaze": args.umaze, "wall": args.wall, "pusht": args.pusht},
        args.output,
    )


if __name__ == "__main__":
    main()

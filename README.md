# Probing Physical Variables in Visual World Models

This repository contains reproducible, notebook-first experiments for asking
where physical variables become linearly readable inside visual world models.
The current demos cover **UMaze**, **Wall**, and **PushT** and measure position,
velocity, acceleration, speed, direction, and object orientation across DINOv2
encoder blocks, learned projected readouts, and dynamics-predictor blocks.

The models are frozen. Each experiment extracts an internal tensor and fits a
small ridge-linear map on training trajectories, then evaluates on unseen
trajectories and unseen spatial regions. A high score means that the variable
is accessible to a linear readout; it does not by itself show that the world
model uses that variable.

## Start with a notebook

| Environment | Demo | Coverage |
|---|---|---|
| UMaze | [Notebook](notebooks/umaze_layerwise_motion_probe_paper_lr.ipynb) | 2,000 unique trajectories |
| Wall | [Notebook](notebooks/wall_layerwise_motion_probe_walkthrough_standalone.ipynb) | 1,920 unique trajectories |
| PushT | [Notebook](notebooks/pusht_layerwise_motion_probe_walkthrough_standalone.ipynb) | 18,500 unique trajectories |

The notebooks are the main demos and include explanations, activation
extraction, probe fitting, controls, uncertainty estimates, and plots. They
clone this repository and download verified assets from releases owned by this
repository or from the original public OSF datasets. They do not clone or read
from the temporal-straightening repository.

## Repository map

```text
notebooks/                 End-to-end Colab demos
scripts/                   Activation extraction and asset utilities
datasets/                  Minimal DINO-WM dataset readers used by the demos
models/                    Minimal DINO-WM runtime needed to load checkpoints
artifacts/checkpoints/     Small configs, checksums, and provenance manifests
results/                   Compact result summaries and selected figures
docs/                      Method and data documentation
tests/                     CPU-only tests for probe construction and assets
```

Large files are intentionally stored as release assets rather than Git
objects. Every downloaded checkpoint, dataset, and activation cache is checked
against a recorded SHA-256 digest. Set `PROBING_VLMS_REPO_URL` to the clone URL
of the anonymous repository and `PROBING_VLMS_RELEASE_BASE` to its
`releases/download` URL before running in Colab. See
[Data and artifacts](docs/data-and-artifacts.md).
For double-blind distribution, also follow the
[anonymous release checklist](docs/anonymous-release.md).

## Core methodology

For a representation at layer `l`:

- position uses one frame feature, `h[t, l]`;
- velocity uses the first temporal difference, `h[t+1, l] - h[t, l]`;
- acceleration uses the second temporal difference,
  `h[t+2, l] - 2 h[t+1, l] + h[t, l]`;
- direction uses unit-vector targets and excludes the slowest training samples;
- PushT orientation uses `(cos(theta), sin(theta))` to avoid an angle wrap.

Each representation is standardized using training statistics only. Ridge
regression uses `lambda=10`. Complete trajectories are divided into 60%
training, 20% validation, and 20% locked test partitions. Representation
selection uses validation only; final evaluation uses the test partition once.
All plotted and headline uncertainty intervals use 1,000 complete-trajectory
bootstrap resamples. See [Methodology](docs/methodology.md).

## Local use

```bash
git clone --branch initial-release <ANONYMOUS_REPOSITORY_URL> probing-VLMs
cd probing-VLMs
python -m pip install -r requirements.txt
```

For notebook and asset setup:

```bash
export PROBING_VLMS_REPO_URL=<ANONYMOUS_REPOSITORY_URL>
export PROBING_VLMS_RELEASE_BASE=<ANONYMOUS_RELEASE_DOWNLOAD_BASE>
```

Checkpoint restoration is normally handled by the notebooks. It can also be
run directly:

```bash
python scripts/fetch_probe_assets.py umaze
python scripts/fetch_probe_assets.py wall
```

These commands download only from `probing-VLMs` releases.

## Result exports

Each notebook writes `validation_selection_scores.csv`,
`headline_selected_test_metrics.csv`,
`headline_straightening_deltas.csv`, and `headline_protocol.json`. After
running all three notebooks, combine their exports with:

```bash
python scripts/collect_headline_exports.py \
  --umaze <UMAZE_OUTPUT_DIR> \
  --wall <WALL_OUTPUT_DIR> \
  --pusht <PUSHT_OUTPUT_DIR>
```

The canonical schemas and protocol manifest are checked into
[`results/`](results/). The collection script refuses empty input tables so
unexecuted notebooks cannot be mistaken for reported measurements.

## Reproducibility notes

- Complete trajectories are split 60/20/20 into training, validation, and a
  locked test partition.
- One deterministic window is selected per trajectory with seed 0.
- OFF and ON conditions use identical windows, labels, splits, and probes.
- Layer/readout selection uses validation trajectories; the selected probe is
  evaluated once on locked test trajectories.
- Headline values use 1,000 complete-trajectory-window bootstrap resamples and
  report 95% percentile intervals.
- Notebook outputs currently cover one frozen training seed per condition;
  bootstrap intervals do not measure training-seed uncertainty.
- UMaze and Wall predictor-action inputs retain the documented legacy-padding
  limitation; encoder and projected-readout results are unaffected.

## Attribution

The minimal model and dataset runtime is adapted from DINO-WM. DINO-WM is
MIT-licensed; DINOv2 and MuJoCo use Apache-2.0. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for exact provenance.

## License

New probe code and documentation are released under the MIT License. Vendored
components retain their upstream notices.

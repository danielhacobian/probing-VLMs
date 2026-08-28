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
| UMaze | [Open in Colab](https://colab.research.google.com/github/danielhacobian/probing-VLMs/blob/initial-release/notebooks/umaze_layerwise_motion_probe_paper_lr.ipynb) | 2,000 unique trajectories |
| Wall | [Open in Colab](https://colab.research.google.com/github/danielhacobian/probing-VLMs/blob/initial-release/notebooks/wall_layerwise_motion_probe_walkthrough_standalone.ipynb) | 1,920 unique trajectories |
| PushT | [Open in Colab](https://colab.research.google.com/github/danielhacobian/probing-VLMs/blob/initial-release/notebooks/pusht_layerwise_motion_probe_walkthrough_standalone.ipynb) | 18,500 unique trajectories |

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

Large files are intentionally stored as GitHub Release assets rather than Git
objects. Every downloaded checkpoint, dataset, and activation cache is checked
against a recorded SHA-256 digest. See [Data and artifacts](docs/data-and-artifacts.md).

## Core methodology

For a representation at layer `l`:

- position uses one frame feature, `h[t, l]`;
- velocity uses the first temporal difference, `h[t+1, l] - h[t, l]`;
- acceleration uses the second temporal difference,
  `h[t+2, l] - 2 h[t+1, l] + h[t, l]`;
- direction uses unit-vector targets and excludes the slowest training samples;
- PushT orientation uses `(cos(theta), sin(theta))` to avoid an angle wrap.

Each representation is standardized using training statistics only. Ridge
regression uses `lambda=10`. Evaluation includes episode holdouts, buffered
spatial holdouts, shuffled labels, position-only controls, position-residualized
targets, and bootstrap intervals. See [Methodology](docs/methodology.md).

## Local use

```bash
git clone --branch initial-release https://github.com/danielhacobian/probing-VLMs.git
cd probing-VLMs
python -m pip install -r requirements.txt
```

Checkpoint restoration is normally handled by the notebooks. It can also be
run directly:

```bash
python scripts/fetch_probe_assets.py umaze
python scripts/fetch_probe_assets.py wall
```

These commands download only from `probing-VLMs` releases.

## Reproducibility notes

- All splits are grouped by complete trajectory.
- One deterministic window is selected per trajectory with seed 0.
- OFF and ON conditions use identical windows, labels, splits, and probes.
- Notebook outputs are descriptive for one frozen checkpoint per condition.
- Bootstrap bands capture held-out-row sampling, not training-seed uncertainty.
- UMaze and Wall predictor-action inputs retain the documented legacy-padding
  limitation; encoder and projected-readout results are unaffected.

## Attribution

The minimal model and dataset runtime is adapted from DINO-WM. DINO-WM is
MIT-licensed; DINOv2 and MuJoCo use Apache-2.0. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for exact provenance.

## License

New probe code and documentation are released under the MIT License. Vendored
components retain their upstream notices.

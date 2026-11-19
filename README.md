# What Does Temporal Straightening Preserve?

This repository contains the code, result exports, and notebook walkthroughs for
**What Does Temporal Straightening Preserve? Matched-Rate Probing of Spatial
State in Visual World Models**. The study asks a narrow question: when temporal
straightening makes latent trajectories geometrically straighter, which physical
variables become easier or harder to recover from the model?

We study DINO-WM-style models in **UMaze**, **Wall**, and **PushT**. Linear ridge
probes measure position, velocity, acceleration, and—where applicable—object
orientation and angular motion across three stages of the model:

1. frozen DINOv2 encoder blocks;
2. the trained projection and aggregation head, where straightening is applied;
3. the causal dynamics predictor, after actions and proprioception have entered.

The main result is not that straightening is uniformly beneficial or harmful.
Its effect depends on the physical variable, environment, and projection learning
rate. A straighter latent trajectory is therefore not automatically a more
readable record of physical state.

## Study design

For each environment, straightening **OFF** and **ON** models are compared at two
matched encoder/projector learning rates, `1e-5` and `1e-6`. The rate is identical
inside each OFF/ON pair. The predictor, action encoder, and proprioceptive encoder
use `5e-4`, while the decoder uses `3e-4`. This gives twelve evaluated checkpoints:

| Environment | Projection rates | OFF/ON pairs | Probe windows |
|---|---:|---:|---:|
| UMaze | `1e-5`, `1e-6` | 2 | 2,000 |
| Wall | `1e-5`, `1e-6` | 2 | 1,920 |
| PushT | `1e-5`, `1e-6` | 2 | 18,500 |

Each example is a four-frame window sampled every five environment steps. One
window is selected from each complete trajectory. Trajectories are split
60/20/20 into probe training, validation, and a locked test set. The layer,
readout, and temporal input are selected on validation trajectories, shared
between OFF and ON, refit on training plus validation data, and evaluated once
on the locked test set.

All matched-rate predictor activations use the complete five-action input expected
by the trained action encoder. No zero-padding is used in the matched-rate study.

## Headline findings

The table below shows selected locked-test predictor results as
`straightening OFF -> ON` in R-squared. These are a compact subset of the full
tables, not a replacement for the committed CSV exports.

| Environment | Rate | Position | Velocity | Object orientation |
|---|---:|---:|---:|---:|
| UMaze | `1e-5` | `.402 -> .786` | `.199 -> .280` | — |
| UMaze | `1e-6` | `.667 -> .729` | `.171 -> .162` | — |
| Wall | `1e-5` | `.685 -> .693` | `.177 -> .220` | — |
| Wall | `1e-6` | `.434 -> .603` | `.165 -> .126` | — |
| PushT | `1e-5` | `.784 -> .399` | `.057 -> .075` | `.466 -> .320` |
| PushT | `1e-6` | `.382 -> .615` | `.045 -> .120` | `.330 -> .209` |

For PushT, the position and velocity columns refer to the block. The clearest
tradeoff is orientation: straightening reduces block-orientation readability at
both rates, even though block position improves at `1e-6`. UMaze position improves
at both rates, while its velocity improvement is clear only at `1e-5`. Wall shows
the same dependence on both variable and learning rate.

Raw DINOv2 CLS and pooled-patch features are identical between an OFF/ON pair
because the backbone is frozen and the input windows are shared. Differences can
first appear in the trained projection/aggregation head and then in the predictor.

The complete locked-test scores, paired ON-minus-OFF differences, validation
selection tables, spatial controls, protocol records, and 95% intervals are in
[`results/matched_rates/`](results/matched_rates/).

## What the probes see

For a layer representation `h[t]`:

- position and orientation probes use the feature from one frame, `h[t]`;
- encoder-side velocity probes may use the first difference,
  `h[t+1] - h[t]`;
- encoder-side acceleration probes may use the second difference,
  `h[t+2] - 2 h[t+1] + h[t]`;
- predictor probes use a single predictor slot, which already contains temporal,
  action, and proprioceptive context;
- PushT orientation is represented as `(cos(theta), sin(theta))` to avoid an
  artificial discontinuity at the angle wrap.

Features are standardized using training statistics only. Every probe is a ridge
regression with `lambda = 10`. Reported intervals use 1,000 paired bootstrap
resamples of complete test-trajectory windows.

See [`docs/methodology.md`](docs/methodology.md) for target construction,
selection, controls, and interpretation details.

## Results and figures

The paper-aligned result package is organized by environment and matched rate:

```text
results/matched_rates/
├── umaze_1e-5/ and umaze_1e-6/
├── wall_1e-5/ and wall_1e-6/
├── pusht_1e-5/ and pusht_1e-6/
├── figures/
├── source_checksums.json
├── verify_results.py
└── plot_results.py
```

Each comparison contains:

- `headline_selected_test_metrics.csv`: validation-selected locked-test scores;
- `headline_straightening_deltas.csv`: paired ON-minus-OFF differences and 95%
  percentile intervals;
- `validation_selection_scores.csv`: the development scores used for selection;
- `matched_layerwise_cartesian_metrics.csv`: layer-wise development curves and
  controls;
- `protocol.json`: model, checkpoint, extraction, and evaluation provenance;
- `split_summary.json`: split counts, disjointness checks, and hashes.

Verify the committed tables and regenerate the paper figures with:

```bash
git clone --branch initial-release https://github.com/danielhacobian/probing-VLMs.git
cd probing-VLMs
python3 -m pip install -r requirements.txt
python3 results/matched_rates/verify_results.py
python3 results/matched_rates/plot_results.py
```

The numerical CSV and JSON files are the source of truth. The plotting script
recreates:

- [`locked_test_deltas.pdf`](results/matched_rates/figures/locked_test_deltas.pdf)
- [`pusht_tradeoff.pdf`](results/matched_rates/figures/pusht_tradeoff.pdf)
- [`depth_and_transfer.pdf`](results/matched_rates/figures/depth_and_transfer.pdf)

## Notebook walkthroughs

The notebooks remain the easiest way to understand the probe construction and
inspect layer-wise curves:

| Environment | Walkthrough | Coverage |
|---|---|---:|
| UMaze | [Open notebook](notebooks/umaze_layerwise_motion_probe_paper_lr.ipynb) | 2,000 trajectories |
| Wall | [Open notebook](notebooks/wall_layerwise_motion_probe_walkthrough_standalone.ipynb) | 1,920 trajectories |
| PushT | [Open notebook](notebooks/pusht_layerwise_motion_probe_walkthrough_standalone.ipynb) | 18,500 trajectories |

These notebooks document the original probe workflow and released-recipe
comparisons. The paper's matched-rate headline results were produced from fresh,
strictly matched activation caches with [`scripts/extract_matched.py`](scripts/extract_matched.py)
and [`scripts/run_matched_probes.py`](scripts/run_matched_probes.py). Use the
committed files under `results/matched_rates/` when citing the matched-rate study.

## Repository map

```text
notebooks/                 End-to-end probe walkthroughs
scripts/                   Activation extraction, matched analysis, and utilities
datasets/                  Minimal DINO-WM dataset readers
models/                    Minimal DINO-WM checkpoint runtime
artifacts/checkpoints/     Configs, checksums, and provenance manifests
results/matched_rates/     Paper-aligned numerical results and figures
docs/                      Method and artifact documentation
tests/                     CPU-only protocol and utility tests
```

Large checkpoints and datasets are stored as release assets or obtained from the
original public DINO-WM datasets rather than committed as Git objects. Downloads
are checked against recorded SHA-256 digests. See
[`docs/data-and-artifacts.md`](docs/data-and-artifacts.md).

## Interpretation limits

- Linear readability means that a restricted linear map can recover a variable;
  it does not show that the model uses that variable causally.
- Predictor features mix visual information with actions and proprioception. The
  predictor is given position and velocity inputs, which affects interpretation.
- Encoder and predictor targets use different temporal alignments, so their
  absolute scores should not be ranked as if they were the same task.
- Spatial holdouts and shortcut controls are development analyses, separate from
  the locked-test headline results.
- Each condition was trained once with seed 0. Bootstrap intervals measure which
  test trajectories were sampled, not uncertainty across model-training seeds.
- The 30 predictor comparisons are not corrected for multiple testing.
- Better probe scores do not establish better planning performance.

## Citation

For the matched-rate study and committed results, please cite:

```bibtex
@article{hacobian2026matched,
  title  = {What Does Temporal Straightening Preserve? Matched-Rate Probing of Spatial State in Visual World Models},
  author = {Hacobian, Daniel and Lee, Erin F. and Sharma, Utsav and
            Dandamudi, Eeshan G. and Mahfoud, Mohammed and Mujumdar, Anusha},
  year   = {2026},
  note   = {Manuscript}
}
```

The notebook-first study that introduced the probe suite appeared at two
non-archival NeurIPS 2026 workshops:

```bibtex
@inproceedings{hacobian2026probing,
  title     = {Probing Physical Readability Under Temporal Straightening in Visual World Models},
  author    = {Hacobian, Daniel and Lee, Erin F. and Sharma, Utsav and
               Dandamudi, Eeshan G. and Mahfoud, Mohammed and Mujumdar, Anusha},
  booktitle = {NeurIPS 2026 Workshop on World Models in Physical AI and
               NeurIPS 2026 Workshop on From Pretrained Representations to Acting Agents},
  year      = {2026},
  note      = {Non-archival workshop paper}
}
```

## Attribution and license

The minimal model and dataset runtime is adapted from DINO-WM. DINO-WM is
MIT-licensed; DINOv2 and MuJoCo use Apache-2.0. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for exact provenance.

New probe code and documentation are released under the MIT License. Vendored
components retain their upstream notices.

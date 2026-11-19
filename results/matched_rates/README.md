# Matched-rate probe results

Six OFF/ON comparisons: UMaze, Wall and PushT at encoder/projector rates
`1e-5` and `1e-6`, training seed 0. Predictor, action and proprioceptive
encoder rates are `5e-4` in both conditions. DINOv2 backbone weights are frozen.

These results are separate from the earlier published-recipe results in the
parent directory. They use newly extracted activations with complete five-action
blocks, not zero-padded actions.

## Files

Each setting contains:

- `headline_selected_test_metrics.csv`: validation-selected locked-test scores.
- `headline_straightening_deltas.csv`: paired ON-minus-OFF differences and intervals.
- `validation_selection_scores.csv`: candidate scores and shared selection.
- `matched_layerwise_cartesian_metrics.csv`: OFF/ON development curves and controls.
- `matched_layerwise_polar_metrics.csv`, where available: OFF/ON polar development
  metrics. UMaze has only the original OFF `layerwise_polar_metrics.csv`; it is
  not a matched ON/OFF export.
- `split_summary.json`: window/partition counts, disjointness checks and SHA-256
  hashes of the original split file and canonical sampled-window/index arrays.
  Full split arrays are omitted; the hashes support comparison after rerunning.
- `protocol.json`: selection, uncertainty, sanitized training configuration and
  checkpoint SHA-256 hashes. A resumed configuration's epoch count may describe
  only the remaining segment; evaluated checkpoints are labeled epoch 20 for
  UMaze/Wall and epoch 2 for PushT.

`source_checksums.json` records hashes checked against the completed-run artifact
manifests and CSV row counts. No checkpoint weights or activation caches are included.

## Recreate the figures

From the repository root, with NumPy, pandas and Matplotlib 3.7+ installed:

```sh
python results/matched_rates/verify_results.py
python results/matched_rates/plot_results.py
```

The plotting script writes three vector PDFs in `figures/`:

- `locked_test_deltas.pdf`: predictor changes with paired 95% intervals.
- `pusht_tradeoff.pdf`: absolute PushT block scores at both rates.
- `depth_and_transfer.pdf`: layer-wise trajectory/spatial **validation** curves,
  not additional locked-test results.

The numerical files are the source of truth. Raw data, checkpoints and activation
extraction are required to rerun the probes, rather than just regenerate plots.
Use the checkpoint hashes and configurations in each `protocol.json` to identify
the evaluated models.

## Interpretation

Probe trajectories are split 60/20/20. One shared candidate per family/target is
selected using mean OFF/ON validation R², then refit on train+validation and scored
on the locked test. Ridge alpha is 10. Intervals use 1,000 paired complete-trajectory
window bootstraps, conditional on one training seed. They are not across-seed
uncertainty estimates or multiplicity-adjusted significance tests.

Predictor features mix visual, action and proprioceptive inputs. DINO temporal
and predictor frame targets have different alignment; their raw scores should not
be ranked as though they were the same task. Spatial controls are development
analyses. Probe-held-out trajectories are not asserted to be world-model-held-out
trajectories. Improved readability is not a demonstrated planning or causal gain.

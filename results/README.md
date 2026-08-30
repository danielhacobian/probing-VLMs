# Results

The notebooks are the authoritative, executable result records. This directory
contains selected figures, the earlier exploratory summary, and four canonical
confirmatory exports:

- `validation_selection_scores.csv` records the shared representation selected
  for each environment, model family, and target. The historical saved outputs
  did not print every candidate's numerical validation score; rerunning the
  notebooks replaces this compact selection record with the full candidate
  table.
- `headline_selected_test_metrics.csv` contains locked-test R² values and 95%
  intervals for every selected OFF/ON representation.
- `headline_straightening_deltas.csv` contains paired ON-minus-OFF R² values
  and 95% intervals.
- `headline_protocol.json` records split sizes, selection policy, bootstrap
  unit, resample count, and available model-training seeds.

After rerunning all three notebooks, `scripts/collect_headline_exports.py`
combines their richer per-environment exports into these canonical files.
Generated activation caches and full exploratory layer tables remain outside
Git because the notebooks reproduce them from verified assets.

Headline findings:

- Position is strongly readable from single-frame features.
- Translational velocity and acceleration become substantially more readable
  from first and second representation differences.
- UMaze heading is robust while scalar speed is weak under the current probes.
- Wall motion transfers to unseen episodes, an unseen spatial region, and the
  held-out doorway region.
- PushT block orientation is readable, while angular velocity and angular
  acceleration remain weak.
- Configuration differences occur in learned projected readouts and predictor
  features, not in the shared frozen raw DINO features.

See `summary.csv` for legacy best episode-held-out OFF scores. Those maxima are
descriptive across evaluated layers/readouts and are not the locked-test
headline table.

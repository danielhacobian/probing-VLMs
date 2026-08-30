# Results

The notebooks are the authoritative, executable result records. This directory
contains selected figures, a legacy descriptive summary, and the canonical
schemas for the confirmatory exports:

- `validation_selection_scores.csv`
- `headline_selected_test_metrics.csv`
- `headline_straightening_deltas.csv`
- `headline_protocol.json`

The three CSVs contain headers until the three notebooks have been rerun under
the current 60/20/20 protocol. This is intentional: the repository does not
present the earlier train/test maxima as locked-test measurements. After the
notebooks finish, use `scripts/collect_headline_exports.py` to replace these
schemas with the combined numerical rows. The script rejects missing or empty
per-environment inputs.

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

See `summary.csv` for legacy best episode-held-out OFF scores from the earlier
exploratory run. Those maxima are retained only for provenance and are not
confirmatory results under the current protocol.

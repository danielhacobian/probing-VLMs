# Results

The notebooks are the authoritative, executable result records. This directory
contains small figures and a compact cross-environment summary suitable for
quick inspection. Generated activation caches and full metric tables are kept
out of Git because the notebooks reproduce them from verified assets.

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

See `summary.csv` for the best episode-held-out OFF scores reported by the
executed notebooks. Maxima are descriptive across evaluated layers/readouts.

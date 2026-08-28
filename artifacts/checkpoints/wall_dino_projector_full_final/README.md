# Final Wall checkpoints

This manifest describes the two epoch-20 checkpoints used for the Wall
layer-wise physical-readability probes.

Each checkpoint is stored in this repository's `probe-checkpoints-v1` Release
as three numbered parts of at most 90 MiB. Run `./restore_checkpoints.sh` from
the repository root to download and reconstruct
`off/model_20.pth` and `on/model_20.pth` and verify their SHA-256 checksums.

- `off/`: no straightening loss
- `on/`: cosine straightening with coefficient 0.1
- `hydra.yaml`: exact training configuration
- `model_20.pth.sha256`: checksum for the reconstructed checkpoint

Only final checkpoints are exposed because intermediate training state is not
needed to reproduce the probing results.

# Artifact manifests

This directory contains only small provenance files: training-configuration
sidecars, expected checkpoint hashes, and restore wrappers. Downloaded model
weights and split parts are ignored by Git.

- UMaze and Wall weights are restored from the `probe-checkpoints-v1` release.
- PushT probe reproduction uses the `pusht-probe-cache-v1` release by default.
- Every reconstructed or downloaded binary is verified before use.

See [`docs/data-and-artifacts.md`](../docs/data-and-artifacts.md) for the full
registry and licensing notes.

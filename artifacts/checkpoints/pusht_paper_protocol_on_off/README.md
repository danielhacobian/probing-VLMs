# PushT paper-protocol checkpoints

This directory records the expected checkpoint bundle for the PushT layerwise
probe notebook. The notebook compares the frozen final straightening-OFF and
straightening-ON models on the same 18,500 public PushT trajectories.

The default notebook path uses verified OFF and ON activation caches from this
repository's `pusht-probe-cache-v1` Release, so checkpoints are not required to
fit probes or reproduce tables and figures. Full activation recomputation
requires the following checkpoint files supplied through `PUSHT_CHECKPOINT_ROOT`:

```text
off/model_latest.pth.part-*
on/model_latest.pth.part-*
```

Set `PUSHT_CHECKPOINT_ROOT` in the notebook to a directory containing verified
`off/model_latest.pth` and `on/model_latest.pth` files.

Expected SHA-256 checksums:

- OFF `model_latest.pth`: `a03cd7e514223db0f3543ce00748036f80df9397dde560684555801e41c936a5`
- ON `model_latest.pth`: `de31f8345d5274cb0dbd68bdaa38e8bab601eb52c4c0f7f83ea0d85a8c20af4c`

The original notebook recorded the following run provenance: two epochs,
seed 0, frame skip 5, three history frames, and one predicted frame. OFF used
encoder learning rate `1e-6` with straightening disabled; ON used encoder
learning rate `1e-5` with `aggcos1e-1` straightening. The YAML sidecars in this
directory reproduce the fields consumed by the notebook.

The repository-owned activation caches are the documented reproducibility path
for the current results; no Google Drive access is required.

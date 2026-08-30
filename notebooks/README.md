# Notebook demos

These notebooks are the primary entry points. Run them from top to bottom in a
fresh Colab GPU runtime. Each setup cell clones the `initial-release` branch,
installs the environment-specific packages, and downloads only the assets that
the selected demo needs.

| Notebook | Default path | Large downloads | Recommended runtime |
|---|---|---|---|
| `umaze_layerwise_motion_probe_paper_lr.ipynb` | Extract fresh OFF and ON activations | OSF UMaze data and two checkpoints | Colab GPU |
| `wall_layerwise_motion_probe_walkthrough_standalone.ipynb` | Extract fresh OFF and ON activations | Wall data release and two checkpoints | Colab GPU |
| `pusht_layerwise_motion_probe_walkthrough_standalone.ipynb` | Refit probes from cached activations | Two release-hosted activation caches | High-memory Colab runtime |

The PushT notebook can recompute activations with `PUSHT_FORCE_RECOMPUTE=1`,
but that optional path requires verified OFF and ON checkpoints supplied through
`PUSHT_CHECKPOINT_ROOT`. The default cache path is the supported standalone
reproduction route and needs no private files or credentials.

Notebook outputs may include paths from the machine on which an executed copy
was produced. The source cells always set the current repository-owned paths.

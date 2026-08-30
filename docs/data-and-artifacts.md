# Data and artifact registry

The demos use public datasets and frozen experimental artifacts. Large files
are not committed to Git. The notebooks download them, verify their SHA-256
digests, and cache them only for the current runtime.

## Public datasets

| Environment | Source | Use |
|---|---|---|
| UMaze | Original DINO-WM OSF archive | 2,000 complete trajectories; one window per trajectory |
| Wall | `wall-probe-data-v1` release in this repository | 1,920 complete trajectories; copied from the public DINO-WM data export |
| PushT | Original DINO-WM OSF archive | 18,500 selected from 18,706 available trajectories when recomputing activations |

The notebooks contain the public OSF URLs and expected hashes. Repository-owned
release URLs are constructed from `PROBING_VLMS_RELEASE_BASE`. The setup cells
provide a working public default, while an anonymous mirror can override the
variable and host the same verified assets.
The original OSF record does not state a separate dataset license, so the
repository does not relicense the trajectories.

## Repository-owned releases

### `probe-checkpoints-v1`

Verified split checkpoint files for UMaze and Wall. The restore utility joins
the parts and checks the final digest recorded under `artifacts/checkpoints/`.

### `wall-probe-data-v1`

The `wall_single.zip` archive used by the Wall notebook.

### `pusht-probe-cache-v1`

OFF and ON activation caches for the matched 18,500 PushT windows. These caches
let the notebook refit all probes and regenerate tables and figures without
redistributing the model checkpoints or re-extracting 42 activation tensors.

## Checkpoint provenance

The checkpoint manifests describe the original training conditions and final
SHA-256 values. They are inputs to this analysis repository, not newly trained
models. Temporal-straightening training code and unrelated planning artifacts
remain outside this repository.

## Adding an asset

1. Place large binaries in a versioned GitHub Release, not in Git history.
2. Record the asset name, source, size, and SHA-256 digest here or in a manifest.
3. Make the downloader fail closed on a digest mismatch.
4. Document whether the asset is required for analysis-only reuse or full
   activation recomputation.

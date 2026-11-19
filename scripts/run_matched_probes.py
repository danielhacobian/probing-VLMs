#!/usr/bin/env python3
"""Execute the repository's analysis cells against newly extracted, matched caches.

No model download, extraction, training, or release-cache fallback is performed.
Use a fresh output directory for every checkpoint pair. Wall requires door XY
coordinates aligned to the cache windows, supplied as an N x 2 .npy file.
"""

from __future__ import annotations
import argparse
import ast
import csv
import gc
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import umaze_probe_walkthrough as shared
from scripts import fast_window_bootstrap

fast_window_bootstrap.install()

NOTEBOOKS = {
    "umaze": "umaze_layerwise_motion_probe_paper_lr.ipynb",
    "wall": "wall_layerwise_motion_probe_walkthrough_standalone.ipynb",
    "pusht": "pusht_layerwise_motion_probe_walkthrough_standalone.ipynb",
}
COUNTS = {"umaze": 2000, "wall": 1920, "pusht": 18500}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checkpoint_backbone_digest(path):
    # Only caller-specified, authorized trained checkpoints are deserialized.
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    h = hashlib.sha256()
    for key, tensor in sorted(payload["encoder"].base_model.state_dict().items()):
        h.update(key.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_namespace(env, notebook):
    ns = dict(
        __name__="matched_probe_analysis",
        np=np,
        plt=plt,
        Path=Path,
        csv=csv,
        json=json,
        os=os,
        sys=sys,
        gc=gc,
        math=math,
        random=random,
        shutil=shutil,
        time=time,
    )
    ns.update({k: v for k, v in vars(shared).items() if not k.startswith("_")})
    source = "".join(notebook["cells"][5]["source"])
    # Load the exact embedded NumPy PushT helper, not its GPU/download setup.
    if env == "pusht":
        for node in ast.parse(source).body:
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "exec"
            ):
                helper = ast.literal_eval(node.value.args[0].args[0])
                exec(compile(helper, "<notebook-pusht-helper>", "exec"), ns)
        # Confirmatory and grouped-bootstrap helpers are shared by all notebooks.
        for node in ast.parse(source).body:
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "scripts.umaze_probe_walkthrough"
            ):
                exec(
                    compile(
                        ast.Module(body=[node], type_ignores=[]),
                        "<shared-import>",
                        "exec",
                    ),
                    ns,
                )
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name in {
            "write_rows",
            "show_rows",
            "grouped",
        }:
            exec(
                compile(
                    ast.Module(body=[node], type_ignores=[]),
                    "<notebook-helper>",
                    "exec",
                ),
                ns,
            )
    return ns


def read_cache(path, ns):
    path = Path(path)
    # Extraction may use portable NPZ for any environment or PushT's mmap directory.
    if path.is_file():
        return shared.load_activation_cache(path)
    if (path / "cache.npz").is_file():
        return shared.load_activation_cache(path / "cache.npz")
    return ns["load_activation_cache"](path)


def validate_pair(pair, env, smoke):
    off, on = pair["off"], pair["on"]
    for i, label in [(1, "states"), (2, "actions"), (3, "choices")]:
        if not np.array_equal(off[i], on[i]):
            raise ValueError(f"OFF/ON {label} differ")
    choices = np.asarray(off[3])
    n = len(choices)
    if choices.shape != (n, 2) or len(np.unique(choices[:, 0])) != n:
        raise ValueError("Require one window per unique trajectory")
    if not smoke and n != COUNTS[env]:
        raise ValueError(f"{env} requires {COUNTS[env]} windows, got {n}")
    if off[1].shape[1] != 4:
        raise ValueError("Require four sampled frames")
    if set(off[0]) != set(on[0]):
        raise ValueError("Representation inventories differ")
    expected = {
        f"dino/{i}/{kind}"
        for i in range(12)
        for kind in ("cls", "pooled_patches", "projected_aggregate")
    }
    expected.update(f"predictor/{i}/pooled_visual" for i in range(6))
    if not smoke and set(off[0]) != expected:
        raise ValueError("Require all 42 repository representation arrays")
    for cond, (reps, states, actions, choices, meta) in pair.items():
        if not meta.get("checkpoint_sha256"):
            raise ValueError(f"{cond} missing checkpoint SHA256")
        if meta.get("frameskip") != 5:
            raise ValueError("Cache must record frameskip=5")
        for key, x in reps.items():
            if len(x) != n or not np.isfinite(x).all():
                raise ValueError(f"Invalid cache array {key}")
            if x.shape != off[0][key].shape:
                raise ValueError(f"Mismatched shape {key}")
    # Exact backbone identity allows small cross-GPU float32 kernel differences.
    hashes = [pair[c][4].get("backbone_sha256") for c in ("off", "on")]
    if all(hashes) and hashes[0] != hashes[1]:
        raise ValueError("Frozen backbone weight hashes differ")
    proven_identical = all(hashes) and hashes[0] == hashes[1]
    # Without exact weight evidence keep original strict numerical tolerance.
    atol = 1e-4 if proven_identical else 1e-5
    # Frozen raw DINO is a paired-input/weight check, not a scientific ON/OFF result.
    for key in off[0]:
        if key.endswith("/cls") or key.endswith("/pooled_patches"):
            if not np.allclose(off[0][key], on[0][key], rtol=1e-4, atol=atol):
                raise ValueError(f"Frozen raw DINO mismatch: {key}")
    return n


def validate_training_configs(pair, learning_rate):
    """Reject mixed-rate or mislabeled checkpoint pairs before fitting."""
    expected = float(learning_rate)
    for condition, straighten in (("off", False), ("on", "aggcos1e-1")):
        training = pair[condition][4].get("training_config", {}).get("training", {})
        if training.get("encoder_lr") != expected:
            raise ValueError(f"{condition}: expected encoder_lr={expected}")
        if training.get("straighten") != straighten:
            raise ValueError(f"{condition}: unexpected straightening configuration")
        if training.get("seed") != 0:
            raise ValueError(f"{condition}: expected training seed 0")


def run(args):
    nbpath = ROOT / "notebooks" / NOTEBOOKS[args.env]
    notebook = json.loads(nbpath.read_text())
    ns = load_namespace(args.env, notebook)
    pair = {c: read_cache(getattr(args, c + "_cache"), ns) for c in ("off", "on")}
    for cond in ("off", "on"):
        checkpoint = getattr(args, cond + "_checkpoint")
        if checkpoint and digest(checkpoint) != pair[cond][4]["checkpoint_sha256"]:
            raise ValueError(f"{cond} checkpoint SHA256 does not match cache")
        if checkpoint and not pair[cond][4].get("backbone_sha256"):
            pair[cond][4]["backbone_sha256"] = checkpoint_backbone_digest(checkpoint)
            pair[cond][4]["backbone_hash_source"] = "verified checkpoint CPU tensors"
    if not args.smoke:
        validate_training_configs(pair, args.learning_rate)
    n = validate_pair(pair, args.env, args.smoke)
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output must be empty to prevent stale table reuse")
    out.mkdir(parents=True, exist_ok=True)
    reps, states, actions, choices, meta = pair["off"]
    onreps, onstates, onactions, onchoices, onmeta = pair["on"]
    provenance = {c: pair[c][4] for c in pair}
    ns.update(
        OUTPUT_DIR=out,
        MATCHED_OUTPUT_DIR=out,
        ROOT=ROOT,
        SEED=0,
        RIDGE=10.0,
        NUM_FRAMES=4,
        FRAME_SKIP=5,
        STEP_DT=1.0,
        BOOTSTRAP_REPEATS=1000,
        SHUFFLE_REPEATS=20,
        MAX_TRAJECTORIES=n,
        MAX_WINDOWS=n,
        TRAJECTORY_LIMIT=n,
        RUN_LABEL=f"{args.env}_matched_lr_{args.learning_rate}_seed0",
        representations=reps,
        states=states,
        actions=actions,
        choices=choices,
        cache_metadata=meta,
        on_representations=onreps,
        on_states=onstates,
        on_actions=onactions,
        on_choices=onchoices,
        on_cache_metadata=onmeta,
        checkpoint_provenance=provenance,
        CHECKPOINTS={
            c: pair[c][4].get("checkpoint", "recorded-by-sha256") for c in pair
        },
    )
    plt.style.use("seaborn-v0_8-whitegrid")
    # Close figures promptly instead of accumulating notebook display objects.
    ns["plt"].show = lambda *a, **kw: plt.close("all")
    manifest = {
        "status": "running",
        "environment": args.env,
        "matched_encoder_learning_rate": args.learning_rate,
        "protocol": "trajectory_grouped_60_20_20_v1",
        "model_training_seeds": [0],
        "notebook": str(nbpath.name),
        "notebook_sha256": digest(nbpath),
        "runner_sha256": digest(__file__),
        "bootstrap_helper_sha256": digest(fast_window_bootstrap.__file__),
        "checkpoint_provenance": provenance,
        "smoke_test_only": args.smoke,
        "raw_dino_numerical_tolerance": {
            "rtol": 1e-4,
            "atol": (
                1e-4 if all(pair[c][4].get("backbone_sha256") for c in pair) else 1e-5
            ),
            "relaxed_only_with_equal_backbone_hashes": True,
        },
        "inference_note": "Linear readability, not causal model use; CIs exclude training-seed variability",
        "adaptations": [
            "additive sufficient-statistics bootstrap; identical RNG draws and 1000 replicates; floating point summation order differs",
            "new checkpoint caches; no release cache fallback",
            "training-consistent action inputs required",
            "Wall doorway buffer uses development data only",
        ],
        "completed_cells": [],
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    def cell(index, source=None):
        s = "".join(notebook["cells"][index]["source"]) if source is None else source
        # All notebook packaging locations become output-local, never /content.
        s = s.replace(
            '"/content/wall_probe_results_confirmatory"',
            'str(OUTPUT_DIR / "wall_probe_results_confirmatory")',
        )
        s = s.replace(
            '"/content/wall_probe_results"', 'str(OUTPUT_DIR / "wall_probe_results")'
        )
        # Archives are built once after confirmatory outputs are present.
        if "archive = shutil.make_archive" in s:
            s = "\n".join(
                l
                for l in s.splitlines()
                if not l.startswith("archive = shutil.make_archive")
            )
            ns["archive"] = "packaged-after-confirmatory"
        if args.smoke:
            s = (
                s.replace("repeats=1000", "repeats=5")
                .replace("repeats=20", "repeats=2")
                .replace(
                    "HEADLINE_BOOTSTRAP_REPEATS = 1000",
                    "HEADLINE_BOOTSTRAP_REPEATS = 5",
                )
            )
            ns["BOOTSTRAP_REPEATS"] = 5
            ns["SHUFFLE_REPEATS"] = 2
        print(f"ANALYSIS_CELL {args.env} {index}", flush=True)
        exec(compile(s, f"{nbpath.name}:cell{index}", "exec"), ns)
        manifest["completed_cells"].append(index)
        (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    if args.env == "wall":
        ns.update(
            representations={c: pair[c][0] for c in pair},
            shared_states=states,
            shared_actions=actions,
            cache_metadata=provenance,
        )
        if not args.door_points:
            raise ValueError("Wall requires --door-points aligned N x 2 .npy")
        door = np.load(args.door_points, allow_pickle=False)
        if door.shape != (n, 2):
            raise ValueError("Invalid door points shape")
        ns["door_points"] = door
        cell(11)
        source = "".join(notebook["cells"][13]["source"])
        start = source.index("door_points = []")
        end = source.index("door_distance =", start)
        source = source[:start] + source[end:]
        source = source.replace(
            "door_distance.max() - door_distance.min()",
            "door_distance[development_idx].max() - door_distance[development_idx].min()",
        )
        cell(13, source)
        for i in [15, 17, 19, 21, 23, 25, 28]:
            cell(i)
    elif args.env == "umaze":
        for i in [9, 11, 13, 15, 17, 19, 21, 23]:
            cell(i)
        ns.update(
            matched_representations={"off": reps, "on": onreps},
            matched_states=states,
            matched_actions=actions,
            matched_choices=choices,
            matched_targets=ns["targets"],
            matched_episode_train=ns["episode_train"],
            matched_episode_validation=ns["episode_validation"],
            matched_episode_test=ns["episode_test"],
            matched_splits=ns["splits"],
        )
        for i in [29, 31, 33, 35]:
            cell(i)
    else:
        for i in [9, 11, 13, 15, 17, 19, 21, 23]:
            cell(i)
        ns.update(
            reuse_on_tables=False,
            matched_cartesian_path=out / "matched_layerwise_cartesian_metrics.csv",
            matched_polar_path=out / "matched_layerwise_polar_metrics.csv",
        )
        for i in [29, 31]:
            cell(i)
        source = "".join(notebook["cells"][33]["source"]).split("package_dir =")[0]
        source = source.replace(
            "recovered runs; encoder learning rates differ",
            "matched encoder/projector learning-rate pair; seed 0",
        )
        cell(33, source)
        cell(35)
    splits = {
        k: np.asarray(ns[k]).tolist()
        for k in ["episode_train", "episode_validation", "episode_test"]
    }
    (out / "trajectory_splits.json").write_text(
        json.dumps({"choices": np.asarray(choices).tolist(), **splits})
    )
    required = [
        "validation_selection_scores.csv",
        "headline_selected_test_metrics.csv",
        "headline_straightening_deltas.csv",
        "headline_protocol.json",
    ]
    for name in required:
        if not (out / name).is_file():
            raise RuntimeError("Missing " + name)
    protocol_path = out / "headline_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol.update(
        smoke_test_only=args.smoke,
        matched_encoder_learning_rate=args.learning_rate,
        checkpoint_provenance=provenance,
    )
    protocol_path.write_text(json.dumps(protocol, indent=2))
    manifest["status"] = "smoke_complete" if args.smoke else "complete"
    manifest["artifacts"] = {
        p.name: digest(p)
        for p in out.iterdir()
        if p.is_file() and p.name != "run_manifest.json"
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", choices=NOTEBOOKS, required=True)
    p.add_argument("--off-cache", required=True)
    p.add_argument("--on-cache", required=True)
    p.add_argument("--learning-rate", "--lr", choices=["1e-5", "1e-6"], required=True)
    p.add_argument("--output", "--output-dir", required=True)
    p.add_argument("--door-points")
    p.add_argument("--off-checkpoint")
    p.add_argument("--on-checkpoint")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Synthetic/reduced validation only, never headline results",
    )
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()

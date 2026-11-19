"""Fresh activation extraction, strict train-consistent action stacking."""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import yaml

from scripts import activation_extraction as ae
from scripts.umaze_probe_walkthrough import save_activation_cache, load_activation_cache
from datasets.img_transforms import default_transform


class Combined:
    def __init__(self, parts):
        self.parts = parts
        self.offsets = np.cumsum([0] + [len(p) for p in parts])
        self.seq_lengths = np.concatenate([np.asarray(p.seq_lengths) for p in parts])

    def locate(self, i):
        j = int(np.searchsorted(self.offsets[1:], i, side="right"))
        return self.parts[j], i - int(self.offsets[j])

    def get_frames(self, i, frames):
        p, j = self.locate(i)
        return p.get_frames(j, frames)

    def action_slice(self, i, start, stop):
        p, j = self.locate(i)
        return p.actions[j, start:stop]


def load_batch(dataset, choices, frameskip, nframes):
    items = [[], [], [], []]
    for ep, start in choices:
        obs, _, state, _ = dataset.get_frames(
            ep, [start + frameskip * i for i in range(nframes)]
        )
        a = (
            dataset.action_slice(ep, start, start + frameskip * nframes)
            if hasattr(dataset, "action_slice")
            else dataset.actions[ep, start : start + frameskip * nframes]
        )
        if len(a) != frameskip * nframes:
            raise ValueError("Window does not contain a complete action stack")
        a = a.reshape(nframes, -1)
        for dest, value in zip(items, [obs["visual"], obs["proprio"], a, state]):
            dest.append(value)
    return tuple(torch.stack(x) for x in items)


def encode_stream(module, values, name):
    expected = module.patch_embed.in_channels
    if values.shape[-1] != expected:
        raise ValueError(
            f"{name}: got {values.shape[-1]} channels, expected {expected}"
        )
    return module(values)


def main(
    env,
    checkpoint,
    output_dir,
    limit=0,
    batch_size=8,
    commit=None,
    *,
    dataset_root,
    training_config=None,
):
    torch.set_num_threads(4)
    assert not (
        Path(output_dir) / "cache.npz"
    ).exists(), "Refusing to overwrite existing extraction"
    root = Path(dataset_root) / env
    if env == "umaze":
        from datasets.point_maze_dset import PointMazeDataset

        dataset = PointMazeDataset(
            data_path=str(root / "point_maze"),
            transform=default_transform(224),
            normalize_action=True,
        )
    elif env == "wall":
        from datasets.wall_dset import WallDataset

        dataset = WallDataset(
            data_path=str(root / "wall_single"),
            transform=default_transform(224),
            normalize_action=True,
        )
        dataset.seq_lengths = np.full(len(dataset), dataset.traj_len)
    else:
        from datasets.pusht_dset import PushTDataset

        parts = sorted(
            p.parent
            for p in root.rglob("states.pth")
            if (p.parent / "seq_lengths.pkl").exists()
        )
        assert parts
        dataset = Combined(
            [
                PushTDataset(
                    data_path=str(p),
                    transform=default_transform(224),
                    normalize_action=True,
                    with_velocity=True,
                )
                for p in parts
            ]
        )
    rng = random.Random(0)
    valid = [i for i, n in enumerate(dataset.seq_lengths) if int(n) >= 20]
    if env == "pusht":
        rng.shuffle(valid)
        valid = sorted(valid[:18500])
    choices = [
        (
            i,
            (rng if env == "umaze" else random.Random(i)).randint(
                0, int(dataset.seq_lengths[i]) - 20
            ),
        )
        for i in valid
    ]
    if limit:
        choices = choices[:limit]
    assert len({i for i, s in choices}) == len(choices)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert len(choices) == (
        {"umaze": 2000, "wall": 1920, "pusht": 18500}[env] if not limit else limit
    )
    modules = ae.load_checkpoint(Path(checkpoint), device)
    for name, module in modules.items():
        assert all(
            torch.isfinite(p).all() for p in module.parameters()
        ), f"nonfinite {name}"
    backbone_hash = hashlib.sha256()
    for key, tensor in sorted(modules["encoder"].base_model.state_dict().items()):
        backbone_hash.update(key.encode())
        backbone_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    with open(checkpoint, "rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    print(
        json.dumps(
            {
                "event": "extraction_start",
                "env": env,
                "checkpoint": checkpoint,
                "sha256": sha,
                "windows": len(choices),
            }
        ),
        flush=True,
    )
    started = time.time()
    extraction_hash = hashlib.sha256(
        Path(__file__).read_bytes() + Path(ae.__file__).read_bytes()
    ).hexdigest()
    part_dir = Path(output_dir) / "parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    chunks = []
    for offset in range(0, len(choices), 256):
        selected = choices[offset : offset + 256]
        part = part_dir / f"{offset:06d}.npz"
        if part.exists():
            r, st, ac, ch, meta = load_activation_cache(part)
            assert (
                meta["checkpoint_sha256"] == sha
                and meta["extraction_code_sha256"] == extraction_hash
                and np.array_equal(ch, selected)
            ), "Invalid resumable part"
        else:
            r, st, ac = ae.collect_activations(
                modules,
                dataset,
                selected,
                batch_size,
                5,
                4,
                device,
                include_kinds={
                    "cls",
                    "pooled_patches",
                    "projected_aggregate",
                    "pooled_visual",
                },
                batch_loader=load_batch,
                stream_encoder=encode_stream,
                antialias=True,
            )
            assert all(np.isfinite(v).all() for v in r.values())
            temporary = part.with_name(part.stem + ".writing.npz")
            save_activation_cache(
                temporary,
                r,
                st,
                ac,
                selected,
                {
                    "checkpoint_sha256": sha,
                    "extraction_code_sha256": extraction_hash,
                    "backbone_sha256": backbone_hash.hexdigest(),
                },
            )
            temporary.rename(part)
            if commit:
                commit()
        chunks.append((r, st, ac))
        print(
            f"COMMITTED_WINDOWS {min(offset+256,len(choices))}/{len(choices)}",
            flush=True,
        )
    reps = {k: np.concatenate([chunk[0][k] for chunk in chunks]) for k in chunks[0][0]}
    states = np.concatenate([chunk[1] for chunk in chunks])
    actions = np.concatenate([chunk[2] for chunk in chunks])
    del chunks
    assert all(np.isfinite(v).all() for v in reps.values())

    config_path = (
        Path(training_config)
        if training_config
        else Path(checkpoint).parents[1] / "hydra.yaml"
    )
    config = yaml.safe_load(config_path.read_text())
    metadata = {
        "extraction_code_sha256": extraction_hash,
        "backbone_sha256": backbone_hash.hexdigest(),
        "training_config": config,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha,
        "env": env,
        "seed": 0,
        "num_frames": 4,
        "frameskip": 5,
        "resize": "bilinear antialias=True to train encoder image size",
        "window_endpoint": "length - frameskip*num_frames, complete last action block",
        "action_encoding": "five contiguous normalized actions per model slot; strict dimensions; no padding",
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "newly_extracted": True,
        "seconds": time.time() - started,
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_activation_cache(
        Path(output_dir) / "cache.npz", reps, states, actions, choices, metadata
    )
    if env == "wall":
        points = []
        for ep, start in choices:
            door = np.asarray(dataset.door_locations[ep, start]).reshape(-1)
            wall = np.asarray(dataset.wall_locations[ep, start]).reshape(-1)
            points.append(door[:2] if door.size >= 2 else [wall[0], door[0]])
        np.save(Path(output_dir) / "door_points.npy", np.asarray(points))
    print(
        json.dumps(
            {
                "event": "extraction_complete",
                "env": env,
                "windows": len(choices),
                "arrays": len(reps),
                "seconds": metadata["seconds"],
                "gpu": metadata["gpu"],
            }
        ),
        flush=True,
    )
    return metadata


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["umaze", "wall", "pusht"], required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--dataset-root", required=True, help="Parent of umaze/, wall/, and pusht/"
    )
    p.add_argument(
        "--training-config",
        help="Hydra config if not stored next to the checkpoint directory",
    )
    main(**vars(p.parse_args()))

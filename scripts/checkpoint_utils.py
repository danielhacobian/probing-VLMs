"""Checkpoint path helpers shared by the activation demos."""

from pathlib import Path


def resolve_checkpoint(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    candidates = [
        path / "checkpoints" / "model_20.pth",
        path / "model_20.pth",
        path / "checkpoints" / "model_latest.pth",
        path / "model_latest.pth",
    ]
    candidates.extend(sorted(path.glob("checkpoints/model_*.pth"), reverse=True))
    candidates.extend(sorted(path.glob("model_*.pth"), reverse=True))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no model checkpoint found under {path}")

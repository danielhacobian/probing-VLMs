#!/usr/bin/env python3
"""Restore frozen probe checkpoints from this repository's GitHub releases.

The probing repository intentionally does not version hundreds of megabytes of
model weights. This script downloads verified sub-100 MB parts from the
``probe-checkpoints-v1`` release, concatenates them, and checks the SHA-256
digest recorded beside each checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = (
    "https://github.com/danielhacobian/probing-VLMs/releases/download/"
    "probe-checkpoints-v1"
)

ASSETS = {
    "umaze": {
        "directory": "umaze_paper_protocol_on_off",
        "filename": "model_20.pth",
        "parts": {
            "off": ["umaze_off_model_20.pth.part-aa", "umaze_off_model_20.pth.part-ab", "umaze_off_model_20.pth.part-ac"],
            "on": ["umaze_on_model_20.pth.part-aa", "umaze_on_model_20.pth.part-ab"],
        },
    },
    "wall": {
        "directory": "wall_dino_projector_full_final",
        "filename": "model_20.pth",
        "parts": {
            "off": ["wall_off_model_20.pth.part-00", "wall_off_model_20.pth.part-01", "wall_off_model_20.pth.part-02"],
            "on": ["wall_on_model_20.pth.part-00", "wall_on_model_20.pth.part-01", "wall_on_model_20.pth.part-02"],
        },
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_digest(checksum_file: Path) -> str:
    return checksum_file.read_text().split()[0]


def restore(environment: str) -> None:
    spec = ASSETS[environment]
    root = ROOT / "artifacts" / "checkpoints" / spec["directory"]
    for condition, suffixes in spec["parts"].items():
        directory = root / condition
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / spec["filename"]
        expected = expected_digest(directory / f"{spec['filename']}.sha256")
        if destination.exists() and sha256(destination) == expected:
            print(f"verified existing {destination}")
            continue

        part_paths = []
        for asset_name in suffixes:
            part_path = directory / asset_name
            if not part_path.exists():
                url = f"{BASE}/{asset_name}"
                print(f"downloading {url}")
                urllib.request.urlretrieve(url, part_path)
            part_paths.append(part_path)

        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("wb") as output:
            for part_path in part_paths:
                with part_path.open("rb") as source:
                    for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        output.write(block)
        if sha256(temporary) != expected:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"checksum mismatch while restoring {destination}")
        temporary.replace(destination)
        print(f"restored and verified {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("environment", choices=sorted(ASSETS))
    restore(parser.parse_args().environment)


if __name__ == "__main__":
    main()

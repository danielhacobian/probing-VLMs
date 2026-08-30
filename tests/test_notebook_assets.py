import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_notebooks_reference_only_the_new_repository():
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert len(notebooks) == 3
    for path in notebooks:
        payload = json.loads(path.read_text())
        text = json.dumps(payload)
        assert "https://github.com/danielhacobian/probing-VLMs" in text
        assert "temporal-straightening.git" not in text
        assert "temporal-straightening/releases" not in text


def test_checkpoint_manifests_exist():
    roots = {
        "umaze_paper_protocol_on_off": "model_20.pth.sha256",
        "wall_dino_projector_full_final": "model_20.pth.sha256",
        "pusht_paper_protocol_on_off": "model_latest.pth.sha256",
    }
    for directory, checksum in roots.items():
        for condition in ("off", "on"):
            assert (
                ROOT / "artifacts" / "checkpoints" / directory / condition / checksum
            ).is_file()

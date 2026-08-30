import csv
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


def test_notebook_confirmatory_protocol_is_consistent():
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        payload = json.loads(path.read_text())
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in payload["cells"]
        )
        assert "episode_group_train_val_test_split" in source
        assert "validation_fraction=0.2, test_fraction=0.2" in source
        assert "HEADLINE_BOOTSTRAP_REPEATS = 1000" in source
        assert "BOOTSTRAP_REPEATS = 300" not in source
        assert "repeats=300" not in source
        assert '"trajectory_split"' in source
        for filename in (
            "validation_selection_scores.csv",
            "headline_selected_test_metrics.csv",
            "headline_straightening_deltas.csv",
            "headline_protocol.json",
        ):
            assert filename in source


def test_committed_headline_exports_are_present_and_nonempty():
    for filename in (
        "validation_selection_scores.csv",
        "headline_selected_test_metrics.csv",
        "headline_straightening_deltas.csv",
    ):
        path = ROOT / "results" / filename
        assert path.is_file()
        with path.open(newline="") as handle:
            assert list(csv.DictReader(handle))

    protocol = json.loads((ROOT / "results" / "headline_protocol.json").read_text())
    assert protocol["protocol_version"] == "trajectory_grouped_60_20_20_v1"
    assert protocol["bootstrap"]["repeats"] == 1000

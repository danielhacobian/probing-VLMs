import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_notebooks_are_anonymous_and_use_configurable_repository_urls():
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert len(notebooks) == 3
    for path in notebooks:
        payload = json.loads(path.read_text())
        text = json.dumps(payload)
        assert not re.search(r"github\.com/[^/]+/probing-VLMs", text, re.I)
        assert not re.search(r"colab\.research\.google\.com/github/", text, re.I)
        assert "PROBING_VLMS_REPO_URL" in text
        assert "PROBING_VLMS_RELEASE_BASE" in text
        assert "temporal-straightening.git" not in text
        assert "temporal-straightening/releases" not in text
        assert "episode_group_train_val_test_split" in text
        assert "trajectory_grouped_60_20_20" in text
        assert "BOOTSTRAP_REPEATS = 300" not in text
        assert "300 held-out bootstrap resamples" not in text
        assert all(not cell.get("outputs") for cell in payload["cells"])
        for export in (
            "validation_selection_scores.csv",
            "headline_selected_test_metrics.csv",
            "headline_straightening_deltas.csv",
            "headline_protocol.json",
        ):
            assert export in text


def test_canonical_headline_exports_and_protocol_exist():
    result_root = ROOT / "results"
    for filename in (
        "validation_selection_scores.csv",
        "headline_selected_test_metrics.csv",
        "headline_straightening_deltas.csv",
    ):
        path = result_root / filename
        assert path.is_file()
        assert path.read_text().strip()

    protocol = json.loads((result_root / "headline_protocol.json").read_text())
    assert protocol["protocol_version"] == "trajectory_grouped_60_20_20_v1"
    assert protocol["selection_split"] == "validation_trajectories"
    assert protocol["evaluation_split"] == "locked_test_trajectories"
    assert protocol["bootstrap_unit"] == "complete_trajectory_window"
    assert protocol["bootstrap_repeats"] == 1000


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

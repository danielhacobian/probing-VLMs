"""Check source hashes, table consistency and recorded probe partition checks."""
import csv
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    manifest = json.loads((HERE / "source_checksums.json").read_text())
    assert len(manifest) == 6, "Expected all six comparisons"
    file_count = 0
    for setting, files in manifest.items():
        folder = HERE / setting
        for name, expected in files.items():
            path = folder / name
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected["sha256"], path
            if "rows" in expected:
                assert len(rows(path)) == expected["rows"], path
            file_count += 1
        protocol = json.loads((folder / "protocol.json").read_text())
        split = json.loads((folder / "split_summary.json").read_text())
        assert split["partitions_disjoint"] and split["partitions_cover_all_windows"]
        assert split["partition_counts"] == protocol["split_windows"]
        assert sum(split["partition_counts"].values()) == split["window_count"]
        scores = rows(folder / "headline_selected_test_metrics.csv")
        index = {(r["condition"], r["family"], r["variable"]): r for r in scores}
        assert len(index) == len(scores), "Duplicate headline keys"
        for delta in rows(folder / "headline_straightening_deltas.csv"):
            off = index[("off", delta["family"], delta["variable"])]
            on = index[("on", delta["family"], delta["variable"])]
            assert off["representation"] == on["representation"]
            assert off["mode"] == on["mode"]
            for condition, score in [("off", off), ("on", on)]:
                value = float(score["r2"])
                assert math.isfinite(value)
                assert math.isclose(value, float(delta[f"{condition}_r2"]), abs_tol=1e-12)
            assert math.isclose(float(on["r2"]) - float(off["r2"]),
                                float(delta["on_minus_off_r2"]), abs_tol=1e-12)
    print(f"Verified {file_count} source files across {len(manifest)} comparisons.")


if __name__ == "__main__":
    main()

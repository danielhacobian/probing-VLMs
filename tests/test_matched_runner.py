"""Fast invariant tests; end-to-end smoke uses --smoke on synthetic NPZs."""

import unittest
import numpy as np
from scripts.run_matched_probes import validate_pair, validate_training_configs


class MatchedCacheValidation(unittest.TestCase):
    def pair(self):
        n = 10
        reps = {"dino/0/cls": np.zeros((n, 4, 3))}
        arrays = (
            reps,
            np.zeros((n, 4, 4)),
            np.zeros((n, 4, 10)),
            np.c_[np.arange(n), np.zeros(n, int)],
            {"checkpoint_sha256": "a", "frameskip": 5},
        )
        other = tuple(v.copy() if hasattr(v, "copy") else v for v in arrays)
        other = ({k: v.copy() for k, v in reps.items()}, *other[1:])
        return {"off": arrays, "on": other}

    def test_smoke_pair(self):
        self.assertEqual(validate_pair(self.pair(), "umaze", True), 10)

    def test_reject_different_actions(self):
        pair = self.pair()
        pair["on"][2][0, 0, 0] = 1
        with self.assertRaisesRegex(ValueError, "actions differ"):
            validate_pair(pair, "umaze", True)

    def test_reject_raw_dino_change(self):
        pair = self.pair()
        pair["on"][0]["dino/0/cls"][0, 0, 0] = 1
        with self.assertRaisesRegex(ValueError, "Frozen raw DINO"):
            validate_pair(pair, "umaze", True)

    def test_small_roundoff_requires_hash(self):
        pair = self.pair()
        pair["on"][0]["dino/0/cls"][0, 0, 0] = 3.2e-5
        with self.assertRaises(ValueError):
            validate_pair(pair, "umaze", True)
        for c in pair:
            pair[c][4]["backbone_sha256"] = "a" * 64
        self.assertEqual(validate_pair(pair, "umaze", True), 10)

    def test_hash_mismatch_rejected(self):
        pair = self.pair()
        pair["off"][4]["backbone_sha256"] = "a" * 64
        pair["on"][4]["backbone_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "weight hashes differ"):
            validate_pair(pair, "umaze", True)

    def test_large_difference_rejected_even_with_hash(self):
        pair = self.pair()
        pair["on"][0]["dino/0/cls"][0, 0, 0] = 0.001
        for c in pair:
            pair[c][4]["backbone_sha256"] = "a" * 64
        with self.assertRaises(ValueError):
            validate_pair(pair, "umaze", True)

    def test_reject_partial_real_run(self):
        with self.assertRaises(ValueError):
            validate_pair(self.pair(), "umaze", False)

    def test_mixed_rate_pair_rejected(self):
        pair = self.pair()
        for condition, mode in (("off", False), ("on", "aggcos1e-1")):
            pair[condition][4]["training_config"] = {
                "training": {"encoder_lr": 1e-5, "seed": 0, "straighten": mode}
            }
        validate_training_configs(pair, "1e-5")
        pair["on"][4]["training_config"]["training"]["encoder_lr"] = 1e-6
        with self.assertRaisesRegex(ValueError, "on: expected encoder_lr"):
            validate_training_configs(pair, "1e-5")


if __name__ == "__main__":
    unittest.main()

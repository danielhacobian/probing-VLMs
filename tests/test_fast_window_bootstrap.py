import unittest
import numpy as np
from scripts import fast_window_bootstrap as fast


class Equivalence(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(90)

    def groups(self, dims=2):
        y = [self.rng.normal(size=(int(n), dims)) for n in self.rng.integers(1, 5, 30)]
        if dims == 1:
            y = [a[:, 0] for a in y]
        p = [a + self.rng.normal(size=a.shape) for a in y]
        return y, p

    def compare(self, y, p, metrics):
        for m in metrics:
            a = fast._original_ci(y, p, m, 100, 42)
            b = fast.trajectory_bootstrap_metric_ci(y, p, m, 100, 42)
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10, equal_nan=True)
            p2 = [a * 0.9 for a in p]
            a = fast._original_paired(y, p, y, p2, m, 100, 42)
            b = fast.paired_trajectory_bootstrap_difference_ci(y, p, y, p2, m, 100, 42)
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10, equal_nan=True)

    def test_vector(self):
        self.compare(*self.groups(), ["r2", "rmse", "mae", "cosine", "angular_mae_deg"])

    def test_scalar(self):
        self.compare(*self.groups(1), ["r2", "rmse", "mae"])

    def test_empty_windows(self):
        y, p = self.groups()
        y[0] = y[0][:0]
        p[0] = p[0][:0]
        self.compare(y, p, ["r2", "rmse", "mae", "cosine", "angular_mae_deg"])

    def test_nan_direction_filter_and_regression_fallback(self):
        y, p = self.groups()
        y[0][0, 0] = np.nan
        p[2][0, 1] = np.nan
        self.compare(y, p, ["r2", "rmse", "mae", "cosine", "angular_mae_deg"])

    def test_large_offset(self):
        y, p = self.groups()
        y = [a + 1e6 for a in y]
        p = [a + 1e6 for a in p]
        self.compare(y, p, ["r2", "rmse", "mae"])

    def test_constant_labels(self):
        y, p = self.groups()
        y = [np.ones_like(a) for a in y]
        self.compare(y, p, ["r2", "rmse", "mae"])


if __name__ == "__main__":
    unittest.main()

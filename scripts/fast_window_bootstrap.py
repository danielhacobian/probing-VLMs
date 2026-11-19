"""Equivalent complete-window bootstrap via additive sufficient statistics.

Keeps RNG draws and percentile intervals unchanged. Floating point summation
order differs from concatenation, so equivalence is numerical, not bitwise.
"""

import numpy as np
from scripts import umaze_probe_walkthrough as shared

_original_ci = shared.trajectory_bootstrap_metric_ci
_original_paired = shared.paired_trajectory_bootstrap_difference_ci


class Unsupported(ValueError):
    pass


def prepare(truth_groups, prediction_groups, metric):
    if len(truth_groups) != len(prediction_groups) or not len(truth_groups):
        raise Unsupported()
    truths = [np.asarray(t, dtype=np.float64) for t in truth_groups]
    preds = [np.asarray(p, dtype=np.float64) for p in prediction_groups]
    if any(t.shape != p.shape or t.ndim not in (1, 2) for t, p in zip(truths, preds)):
        raise Unsupported()
    if metric in ("r2", "rmse", "mae"):
        if any(not np.isfinite(x).all() for x in truths + preds):
            raise Unsupported()
        arrays = [
            (
                t.reshape(len(t), -1)
                if len(t)
                else t.reshape(0, 1 if t.ndim == 1 else t.shape[1])
            )
            for t in truths
        ]
        if not sum(len(t) for t in arrays):
            raise Unsupported()
        d = arrays[0].shape[1]
        if any(t.shape[1] != d for t in arrays):
            raise Unsupported()
        center = np.concatenate(arrays).mean(axis=0)
        rows = []
        for t, p, a in zip(truths, preds, arrays):
            z = a - center
            error = t - p
            rows.append(
                np.r_[
                    len(t),
                    error.size,
                    np.square(error).sum(),
                    np.abs(error).sum(),
                    np.square(z).sum(),
                    z.sum(axis=0),
                ]
            )
        stats = np.asarray(rows)

        def score(sample):
            s = stats[sample].sum(axis=0)
            if metric == "r2":
                return 1.0 - s[2] / max(s[4] - np.square(s[5:]).sum() / s[0], 1e-12)
            if metric == "rmse":
                return np.sqrt(s[2] / s[1])
            return s[3] / s[1]

        return score
    if metric in ("cosine", "angular_mae_deg"):
        rows = []
        for t, p in zip(truths, preds):
            if t.ndim != 2 or t.shape[1] != 2:
                raise Unsupported()
            valid = np.isfinite(t).all(axis=-1) & np.isfinite(p).all(axis=-1)
            t = t[valid]
            p = p[valid]
            if metric == "cosine":
                values = np.sum(t * p, axis=-1) / (
                    np.linalg.norm(t, axis=-1) * np.linalg.norm(p, axis=-1) + 1e-8
                )
            else:
                theta = np.arctan2(p[:, 1], p[:, 0]) - np.arctan2(t[:, 1], t[:, 0])
                values = np.degrees(np.abs(np.arctan2(np.sin(theta), np.cos(theta))))
            rows.append([len(values), values.sum()])
        stats = np.asarray(rows)

        def score(sample):
            s = stats[sample].sum(axis=0)
            return s[1] / s[0] if s[0] else float("nan")

        return score
    raise Unsupported()


def trajectory_bootstrap_metric_ci(
    truth_groups, prediction_groups, metric="r2", repeats=1000, seed=0
):
    try:
        score = prepare(truth_groups, prediction_groups, metric)
    except Unsupported:
        return _original_ci(truth_groups, prediction_groups, metric, repeats, seed)
    rng = np.random.default_rng(seed)
    n = len(truth_groups)
    values = [score(rng.integers(0, n, n)) for _ in range(repeats)]
    return np.quantile(values, [0.025, 0.975]).tolist()


def paired_trajectory_bootstrap_difference_ci(
    first_truth_groups,
    first_prediction_groups,
    second_truth_groups,
    second_prediction_groups,
    metric="r2",
    repeats=1000,
    seed=0,
):
    try:
        if len(first_truth_groups) != len(second_truth_groups):
            raise Unsupported()
        first = prepare(first_truth_groups, first_prediction_groups, metric)
        second = prepare(second_truth_groups, second_prediction_groups, metric)
    except Unsupported:
        return _original_paired(
            first_truth_groups,
            first_prediction_groups,
            second_truth_groups,
            second_prediction_groups,
            metric,
            repeats,
            seed,
        )
    rng = np.random.default_rng(seed)
    n = len(first_truth_groups)
    values = []
    for _ in range(repeats):
        sampled = rng.integers(0, n, n)
        values.append(second(sampled) - first(sampled))
    return np.quantile(values, [0.025, 0.975]).tolist()


def install():
    shared.trajectory_bootstrap_metric_ci = trajectory_bootstrap_metric_ci
    shared.paired_trajectory_bootstrap_difference_ci = (
        paired_trajectory_bootstrap_difference_ci
    )

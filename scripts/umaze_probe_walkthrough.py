#!/usr/bin/env python3
"""Reusable analysis helpers for the UMaze layerwise probe walkthrough.

The expensive model-specific activation collection remains in
``probe_umaze_layers.py``.  This module contains NumPy-only alignment, split,
linear-probe, control, cache, and uncertainty utilities so the notebook can be
tested without a checkpoint or GPU.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def r2_score(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    denominator = np.square(y - y.mean(axis=0)).sum()
    return float(1.0 - np.square(y - prediction).sum() / max(denominator, 1e-12))


def standardize_fit(x_train, y_train, x_test, ridge):
    """NumPy ridge implementation matching ``probe_umaze_layers.py``."""
    x_mean = x_train.mean(0, keepdims=True)
    x_std = x_train.std(0, keepdims=True)
    x_std[x_std < 1e-6] = 1.0
    xs = (x_train - x_mean) / x_std
    xt = (x_test - x_mean) / x_std
    y = np.asarray(y_train)
    if y.ndim == 1:
        y = y[:, None]
    y_mean = y.mean(0, keepdims=True)
    yc = y - y_mean
    n, d = xs.shape
    if d <= n:
        weight = np.linalg.solve(xs.T @ xs + ridge * np.eye(d), xs.T @ yc)
    else:
        weight = xs.T @ np.linalg.solve(xs @ xs.T + ridge * np.eye(n), yc)
    prediction = xt @ weight + y_mean
    return (
        prediction.squeeze(-1) if prediction.shape[1] == 1 else prediction,
        weight,
    )


def build_motion_targets(states: np.ndarray, frameskip: int, step_dt: float = 1.0):
    """Create framewise and transitionwise Cartesian/polar motion targets.

    State channels 2:4 are used as instantaneous velocity when available.
    Transition velocity always comes from displacement, making it the matching
    label for a temporal feature difference spanning two sampled frames.
    """
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 3 or states.shape[-1] < 2:
        raise ValueError("states must have shape [window, time, >=2]")
    dt = float(frameskip) * float(step_dt)
    if dt <= 0:
        raise ValueError("frameskip * step_dt must be positive")

    position = states[..., :2]
    transition_velocity = np.diff(position, axis=1) / dt
    if states.shape[-1] >= 4:
        velocity = states[..., 2:4]
    else:
        velocity = np.empty_like(position)
        velocity[:, :-1] = transition_velocity
        velocity[:, -1] = transition_velocity[:, -1]

    acceleration = np.full_like(velocity, np.nan)
    acceleration[:, 1:] = np.diff(velocity, axis=1) / dt
    transition_acceleration = np.diff(transition_velocity, axis=1) / dt

    def polar(vector):
        magnitude = np.linalg.norm(vector, axis=-1)
        direction = vector / np.maximum(magnitude[..., None], 1e-8)
        return magnitude, direction

    speed, heading = polar(velocity)
    transition_speed, transition_heading = polar(transition_velocity)
    acceleration_magnitude, acceleration_direction = polar(np.nan_to_num(acceleration))
    acceleration_direction[:, 0] = np.nan
    transition_acceleration_magnitude, transition_acceleration_direction = polar(
        transition_acceleration
    )
    return {
        "position": position,
        "velocity": velocity,
        "speed": speed,
        "heading": heading,
        "acceleration": acceleration,
        "acceleration_magnitude": acceleration_magnitude,
        "acceleration_direction": acceleration_direction,
        "transition_velocity": transition_velocity,
        "transition_speed": transition_speed,
        "transition_heading": transition_heading,
        "transition_acceleration": transition_acceleration,
        "transition_acceleration_magnitude": transition_acceleration_magnitude,
        "transition_acceleration_direction": transition_acceleration_direction,
        "dt": dt,
    }


def align_representation(rep: np.ndarray, targets: dict, variable: str, mode: str):
    """Align a representation tensor and physical target without flattening windows.

    Returns ``(features, labels, position_context)``.  Position context is the
    location matched to each label and supports position-only and residualized
    controls.
    """
    rep = np.asarray(rep)
    if rep.ndim == 4:
        rep = rep.mean(axis=2)
    if rep.ndim != 3:
        raise ValueError("representation must have shape [window, time, feature]")
    t = rep.shape[1]
    position = targets["position"][:, :t]

    frame_targets = {
        "position": targets["position"][:, :t],
        "velocity": targets["velocity"][:, :t],
        "speed": targets["speed"][:, :t],
        "heading": targets["heading"][:, :t],
        "acceleration": targets["acceleration"][:, :t],
        "acceleration_magnitude": targets["acceleration_magnitude"][:, :t],
        "acceleration_direction": targets["acceleration_direction"][:, :t],
    }
    transition_targets = {
        "velocity": targets["transition_velocity"][:, : max(t - 1, 0)],
        "speed": targets["transition_speed"][:, : max(t - 1, 0)],
        "heading": targets["transition_heading"][:, : max(t - 1, 0)],
    }
    second_targets = {
        "acceleration": targets["transition_acceleration"][:, : max(t - 2, 0)],
        "acceleration_magnitude": targets["transition_acceleration_magnitude"][:, : max(t - 2, 0)],
        "acceleration_direction": targets["transition_acceleration_direction"][:, : max(t - 2, 0)],
    }

    if mode == "frame":
        if variable not in frame_targets:
            raise ValueError(f"{variable!r} has no framewise target")
        return rep, frame_targets[variable], position
    if mode in ("delta", "concat"):
        if variable not in transition_targets:
            raise ValueError(f"{mode} is only defined for velocity/speed/heading")
        if t < 2:
            raise ValueError("at least two representation slots are required")
        features = np.diff(rep, axis=1) if mode == "delta" else np.concatenate(
            [rep[:, :-1], rep[:, 1:]], axis=-1
        )
        context = 0.5 * (position[:, :-1] + position[:, 1:])
        return features, transition_targets[variable], context
    if mode in ("second_delta", "concat3"):
        if variable not in second_targets:
            raise ValueError(f"{mode} is only defined for acceleration targets")
        if t < 3:
            raise ValueError("at least three representation slots are required")
        features = (
            rep[:, 2:] - 2.0 * rep[:, 1:-1] + rep[:, :-2]
            if mode == "second_delta"
            else np.concatenate([rep[:, :-2], rep[:, 1:-1], rep[:, 2:]], axis=-1)
        )
        return features, second_targets[variable], position[:, 1:-1]
    raise ValueError(f"unknown mode {mode!r}")


def episode_group_split(choices, test_fraction: float = 0.2, seed: int = 0):
    """Split windows by episode so one trajectory never appears on both sides."""
    choices = np.asarray(choices, dtype=np.int64)
    if choices.ndim != 2 or choices.shape[1] < 1:
        raise ValueError("choices must contain [episode, start] rows")
    episodes = np.unique(choices[:, 0])
    if len(episodes) < 2:
        raise ValueError("episode-held-out evaluation requires at least two episodes")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(episodes)
    n_test = min(len(episodes) - 1, max(1, int(math.ceil(test_fraction * len(episodes)))))
    test_episodes = set(shuffled[:n_test].tolist())
    test = np.asarray([i for i, episode in enumerate(choices[:, 0]) if episode in test_episodes])
    train = np.asarray([i for i, episode in enumerate(choices[:, 0]) if episode not in test_episodes])
    return train, test


def episode_group_train_val_test_split(
    choices,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 0,
):
    """Make disjoint train/validation/test partitions of complete trajectories.

    The permutation is deterministic. Validation episodes are taken first and
    test episodes second, so the validation partition matches the historical
    80/20 episode holdout when both fractions are 0.2. No episode can appear in
    more than one partition, even when there are multiple windows per episode.
    """
    choices = np.asarray(choices, dtype=np.int64)
    if choices.ndim != 2 or choices.shape[1] < 1:
        raise ValueError("choices must contain [episode, start] rows")
    if not 0.0 < validation_fraction < 1.0 or not 0.0 < test_fraction < 1.0:
        raise ValueError("validation_fraction and test_fraction must be in (0, 1)")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation and test fractions must leave training episodes")
    episodes = np.unique(choices[:, 0])
    if len(episodes) < 3:
        raise ValueError("train/validation/test evaluation requires at least three episodes")
    shuffled = np.random.default_rng(seed).permutation(episodes)
    n_validation = max(1, int(math.ceil(validation_fraction * len(episodes))))
    n_test = max(1, int(math.ceil(test_fraction * len(episodes))))
    if n_validation + n_test >= len(episodes):
        n_test = 1
        n_validation = 1
    validation_episodes = set(shuffled[:n_validation].tolist())
    test_episodes = set(shuffled[n_validation : n_validation + n_test].tolist())
    train_episodes = set(shuffled[n_validation + n_test :].tolist())

    def rows_for(group):
        return np.asarray(
            [i for i, episode in enumerate(choices[:, 0]) if episode in group],
            dtype=np.int64,
        )

    train, validation, test = (
        rows_for(train_episodes),
        rows_for(validation_episodes),
        rows_for(test_episodes),
    )
    if not len(train) or not len(validation) or not len(test):
        raise ValueError("trajectory split produced an empty partition")
    return train, validation, test


def spatial_holdout_split(
    anchor_position: np.ndarray,
    axis: int = 1,
    quantile: float = 0.8,
    high: bool = True,
    buffer_fraction: float = 0.05,
):
    """Hold out one spatial tail and drop a buffer band around its boundary."""
    anchor_position = np.asarray(anchor_position)
    coordinate = anchor_position[:, axis]
    boundary = float(np.quantile(coordinate, quantile if high else 1.0 - quantile))
    buffer = float(np.ptp(coordinate) * buffer_fraction)
    if high:
        train = np.flatnonzero(coordinate < boundary - buffer)
        test = np.flatnonzero(coordinate >= boundary)
    else:
        train = np.flatnonzero(coordinate > boundary + buffer)
        test = np.flatnonzero(coordinate <= boundary)
    if not len(train) or not len(test):
        raise ValueError("spatial split produced an empty train or test set")
    return train, test, {"boundary": boundary, "buffer": buffer, "axis": axis, "high": high}


def _flatten_valid(features, labels, window_indices):
    x = np.asarray(features)[window_indices].reshape(-1, features.shape[-1])
    y = np.asarray(labels)[window_indices]
    y = y.reshape(-1, *y.shape[2:])
    finite = np.isfinite(x).all(axis=-1)
    finite &= np.isfinite(y).all(axis=-1) if y.ndim > 1 else np.isfinite(y)
    return x[finite], y[finite]


def fit_probe(features, labels, train_idx, test_idx, ridge: float = 10.0):
    """Fit the standardized ridge probe used by the existing UMaze analysis."""
    x_train, y_train = _flatten_valid(features, labels, train_idx)
    x_test, y_test = _flatten_valid(features, labels, test_idx)
    prediction, weight = standardize_fit(x_train, y_train, x_test, ridge)
    if y_test.ndim == 2 and y_test.shape[1] == 1 and prediction.ndim == 1:
        y_test = y_test[:, 0]
    return y_test, prediction, weight


def regression_scores(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    return {
        "r2": r2_score(truth, prediction),
        "rmse": float(np.sqrt(np.mean(np.square(truth - prediction)))),
        "mae": float(np.mean(np.abs(truth - prediction))),
    }


def direction_scores(truth, prediction, minimum_magnitude=None):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    mask = np.isfinite(truth).all(axis=-1) & np.isfinite(prediction).all(axis=-1)
    if minimum_magnitude is not None:
        mask &= np.linalg.norm(truth, axis=-1) >= minimum_magnitude
    truth, prediction = truth[mask], prediction[mask]
    cosine = np.sum(truth * prediction, axis=-1) / (
        np.linalg.norm(truth, axis=-1) * np.linalg.norm(prediction, axis=-1) + 1e-8
    )
    true_theta = np.arctan2(truth[:, 1], truth[:, 0])
    pred_theta = np.arctan2(prediction[:, 1], prediction[:, 0])
    delta = np.arctan2(np.sin(pred_theta - true_theta), np.cos(pred_theta - true_theta))
    return {
        "cosine": float(np.mean(cosine)),
        "angular_mae_deg": float(np.degrees(np.mean(np.abs(delta)))),
    }


def mask_slow_directions(labels, magnitude, train_idx, quantile: float = 0.1):
    """Mark low-magnitude direction labels NaN using a training-only cutoff."""
    labels = np.asarray(labels, dtype=float).copy()
    magnitude = np.asarray(magnitude)
    cutoff = float(np.nanquantile(magnitude[train_idx], quantile))
    labels[magnitude <= max(cutoff, 1e-8)] = np.nan
    return labels, cutoff


def shuffled_label_score(
    features,
    labels,
    train_idx,
    test_idx,
    ridge: float = 10.0,
    repeats: int = 20,
    seed: int = 0,
):
    """Return R² values after shuffling labels at the window level."""
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        shuffled = np.asarray(labels).copy()
        shuffled[train_idx] = shuffled[rng.permutation(train_idx)]
        truth, prediction, _ = fit_probe(features, shuffled, train_idx, test_idx, ridge)
        values.append(r2_score(truth, prediction))
    return np.asarray(values)


def residualize_against_position(labels, position_context, train_idx, ridge: float = 10.0):
    """Subtract the component predictable from XY using training windows only."""
    labels = np.asarray(labels)
    position_context = np.asarray(position_context)
    x_train, y_train = _flatten_valid(position_context, labels, train_idx)
    all_idx = np.arange(len(labels))
    x_all = position_context.reshape(-1, position_context.shape[-1])
    finite_all = np.isfinite(x_all).all(axis=-1)
    prediction = np.full((len(x_all),) + labels.shape[2:], np.nan, dtype=float)
    pred_valid, _ = standardize_fit(x_train, y_train, x_all[finite_all], ridge)
    if prediction.ndim == 2 and prediction.shape[1] == 1 and pred_valid.ndim == 1:
        pred_valid = pred_valid[:, None]
    prediction[finite_all] = pred_valid
    prediction = prediction.reshape(labels.shape)
    return labels - prediction


def bootstrap_metric_ci(truth, prediction, metric="r2", repeats=1000, seed=0):
    """Bootstrap rows of a held-out prediction and return a percentile interval."""
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        index = rng.integers(0, len(truth), len(truth))
        if metric == "r2":
            values.append(r2_score(truth[index], prediction[index]))
        elif metric == "cosine":
            values.append(direction_scores(truth[index], prediction[index])["cosine"])
        else:
            raise ValueError(f"unsupported metric {metric!r}")
    return np.quantile(values, [0.025, 0.975]).tolist()


def fit_probe_grouped(features, labels, train_idx, evaluation_idx, ridge: float = 10.0):
    """Fit once and return predictions grouped by complete evaluation windows.

    Each list element contains all valid temporal rows from one trajectory
    window. Keeping this boundary intact lets the bootstrap resample complete
    windows instead of treating adjacent frames as independent observations.
    """
    features = np.asarray(features)
    labels = np.asarray(labels)
    x_train, y_train = _flatten_valid(features, labels, np.asarray(train_idx))
    x_mean = x_train.mean(0, keepdims=True)
    x_std = x_train.std(0, keepdims=True)
    x_std[x_std < 1e-6] = 1.0
    xs = (x_train - x_mean) / x_std
    y = y_train[:, None] if y_train.ndim == 1 else y_train
    y_mean = y.mean(0, keepdims=True)
    yc = y - y_mean
    n, d = xs.shape
    if d <= n:
        weight = np.linalg.solve(xs.T @ xs + ridge * np.eye(d), xs.T @ yc)
    else:
        weight = xs.T @ np.linalg.solve(xs @ xs.T + ridge * np.eye(n), yc)

    truth_groups, prediction_groups = [], []
    for window in np.asarray(evaluation_idx, dtype=np.int64):
        x = features[window].reshape(-1, features.shape[-1])
        target = labels[window]
        target = target.reshape(-1, *target.shape[1:])
        finite = np.isfinite(x).all(axis=-1)
        finite &= (
            np.isfinite(target).all(axis=-1)
            if target.ndim > 1
            else np.isfinite(target)
        )
        if not finite.any():
            continue
        prediction = ((x[finite] - x_mean) / x_std) @ weight + y_mean
        prediction = prediction[:, 0] if target[finite].ndim == 1 else prediction
        truth_groups.append(target[finite])
        prediction_groups.append(prediction)
    if not truth_groups:
        raise ValueError("evaluation partition contains no finite target rows")
    return truth_groups, prediction_groups, weight


def group_flat_predictions_by_window(
    features,
    labels,
    evaluation_idx,
    truth,
    prediction,
):
    """Recover complete-window groups from aligned flattened predictions."""
    features = np.asarray(features)
    labels = np.asarray(labels)
    truth = np.asarray(truth)
    prediction = np.asarray(prediction)
    truth_groups, prediction_groups = [], []
    offset = 0
    for window in np.asarray(evaluation_idx, dtype=np.int64):
        x = features[window].reshape(-1, features.shape[-1])
        target = labels[window].reshape(-1, *labels[window].shape[1:])
        finite = np.isfinite(x).all(axis=-1)
        finite &= (
            np.isfinite(target).all(axis=-1)
            if target.ndim > 1
            else np.isfinite(target)
        )
        count = int(finite.sum())
        if count:
            truth_groups.append(truth[offset : offset + count])
            prediction_groups.append(prediction[offset : offset + count])
            offset += count
    if offset != len(truth) or offset != len(prediction):
        raise ValueError("flattened predictions do not match complete-window validity masks")
    return truth_groups, prediction_groups


def grouped_metric(truth_groups, prediction_groups, metric="r2", group_indices=None):
    """Score concatenated rows after selecting whole trajectory windows."""
    if group_indices is None:
        group_indices = np.arange(len(truth_groups))
    truth = np.concatenate([truth_groups[int(i)] for i in group_indices], axis=0)
    prediction = np.concatenate(
        [prediction_groups[int(i)] for i in group_indices], axis=0
    )
    if metric == "r2":
        return r2_score(truth, prediction)
    if metric == "rmse":
        return float(np.sqrt(np.mean(np.square(truth - prediction))))
    if metric == "mae":
        return float(np.mean(np.abs(truth - prediction)))
    if metric == "cosine":
        return direction_scores(truth, prediction)["cosine"]
    if metric == "angular_mae_deg":
        return direction_scores(truth, prediction)["angular_mae_deg"]
    raise ValueError(f"unsupported grouped metric {metric!r}")


def trajectory_bootstrap_metric_ci(
    truth_groups,
    prediction_groups,
    metric="r2",
    repeats: int = 1000,
    seed: int = 0,
):
    """Return a 95% percentile interval from complete-window resamples."""
    if len(truth_groups) != len(prediction_groups) or not truth_groups:
        raise ValueError("truth and prediction groups must be non-empty and aligned")
    rng = np.random.default_rng(seed)
    count = len(truth_groups)
    values = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        sampled = rng.integers(0, count, count)
        values[repeat] = grouped_metric(
            truth_groups, prediction_groups, metric, sampled
        )
    return np.quantile(values, [0.025, 0.975]).tolist()


def grouped_regression_summary(
    truth_groups,
    prediction_groups,
    repeats: int = 1000,
    seed: int = 0,
):
    """Point estimates and trajectory-bootstrap intervals for headline values."""
    result = {"test_trajectories": len(truth_groups)}
    for offset, metric in enumerate(("r2", "rmse", "mae")):
        result[metric] = grouped_metric(truth_groups, prediction_groups, metric)
        low, high = trajectory_bootstrap_metric_ci(
            truth_groups,
            prediction_groups,
            metric,
            repeats=repeats,
            seed=seed + offset,
        )
        result[f"{metric}_ci_low"] = low
        result[f"{metric}_ci_high"] = high
    return result


def paired_trajectory_bootstrap_difference_ci(
    first_truth_groups,
    first_prediction_groups,
    second_truth_groups,
    second_prediction_groups,
    metric="r2",
    repeats: int = 1000,
    seed: int = 0,
):
    """Bootstrap a paired second-minus-first metric difference by trajectory."""
    counts = {
        len(first_truth_groups), len(first_prediction_groups),
        len(second_truth_groups), len(second_prediction_groups),
    }
    if len(counts) != 1 or not first_truth_groups:
        raise ValueError("paired trajectory groups must be non-empty and aligned")
    count = counts.pop()
    rng = np.random.default_rng(seed)
    values = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        sampled = rng.integers(0, count, count)
        first = grouped_metric(
            first_truth_groups, first_prediction_groups, metric, sampled
        )
        second = grouped_metric(
            second_truth_groups, second_prediction_groups, metric, sampled
        )
        values[repeat] = second - first
    return np.quantile(values, [0.025, 0.975]).tolist()


def select_then_test_representations(
    representations_by_condition,
    targets,
    align_fn,
    specs_fn,
    train_idx,
    validation_idx,
    test_idx,
    ridge: float = 10.0,
    bootstrap_repeats: int = 1000,
    seed: int = 0,
    model_seed: int = 0,
):
    """Select layer/readout on validation trajectories, then test it once.

    ``specs_fn(family, representation)`` supplies the target/mode candidates
    valid for a representation. One shared candidate is selected for each
    representation family and target by averaging its validation R² across
    conditions. The same layer, readout, and temporal construction is therefore
    tested for OFF and ON. Test labels are not accessed until all selections are
    fixed. Returned test intervals resample complete trajectory windows, and
    OFF/ON differences use paired trajectory resamples.
    """
    validation_rows = []
    aligned = {}
    for condition, representations in representations_by_condition.items():
        for name, representation in sorted(representations.items()):
            family, layer, kind = name.split("/", 2)
            for variable, mode in specs_fn(family, representation):
                features, labels, _ = align_fn(
                    representation, targets, variable, mode
                )
                truth_groups, prediction_groups, _ = fit_probe_grouped(
                    features, labels, train_idx, validation_idx, ridge
                )
                row = {
                    "condition": condition,
                    "model_seed": model_seed,
                    "family": family,
                    "variable": variable,
                    "representation": name,
                    "layer": int(layer),
                    "kind": kind,
                    "mode": mode,
                    "validation_r2": grouped_metric(
                        truth_groups, prediction_groups, "r2"
                    ),
                    "validation_trajectories": len(truth_groups),
                }
                validation_rows.append(row)
                aligned[(condition, name, variable, mode)] = (features, labels)

    conditions = tuple(sorted(representations_by_condition))
    candidate_groups = {}
    for row in validation_rows:
        key = (
            row["family"], row["variable"], row["representation"], row["mode"]
        )
        candidate_groups.setdefault(key, []).append(row)
    shared_winners = {}
    for candidate_key, rows in candidate_groups.items():
        if {row["condition"] for row in rows} != set(conditions):
            continue
        family, variable, _, _ = candidate_key
        mean_score = float(np.mean([row["validation_r2"] for row in rows]))
        winner_key = (family, variable)
        if (
            winner_key not in shared_winners
            or mean_score > shared_winners[winner_key][0]
        ):
            shared_winners[winner_key] = (mean_score, rows)
    selected = {}
    selected_candidates = set()
    for (family, variable), (mean_score, rows) in shared_winners.items():
        selected_candidates.add(
            (family, variable, rows[0]["representation"], rows[0]["mode"])
        )
        for row in rows:
            selected[(row["condition"], family, variable)] = {
                **row,
                "selection_mean_validation_r2": mean_score,
                "selection_scope": "shared_across_conditions",
            }
    for row in validation_rows:
        row["selected"] = (
            row["family"], row["variable"], row["representation"], row["mode"]
        ) in selected_candidates

    development_idx = np.sort(
        np.concatenate([np.asarray(train_idx), np.asarray(validation_idx)])
    )
    headline_rows = []
    test_groups = {}
    for key, selection in sorted(selected.items()):
        condition, family, variable = key
        features, labels = aligned[
            (condition, selection["representation"], variable, selection["mode"])
        ]
        truth_groups, prediction_groups, _ = fit_probe_grouped(
            features, labels, development_idx, test_idx, ridge
        )
        summary = grouped_regression_summary(
            truth_groups,
            prediction_groups,
            repeats=bootstrap_repeats,
            seed=seed,
        )
        headline_rows.append({
            **{
                name: value for name, value in selection.items()
                if name not in (
                    "validation_r2",
                    "validation_trajectories",
                    "selection_mean_validation_r2",
                )
            },
            **summary,
            "selection_split": "validation_trajectories",
            "evaluation_split": "locked_test_trajectories",
            "bootstrap_unit": "complete_trajectory_window",
            "interval": "95_percentile",
        })
        test_groups[key] = (truth_groups, prediction_groups)

    delta_rows = []
    comparable = sorted({
        (row["family"], row["variable"])
        for row in headline_rows
        if ("off", row["family"], row["variable"]) in test_groups
        and ("on", row["family"], row["variable"]) in test_groups
    })
    lookup = {
        (row["condition"], row["family"], row["variable"]): row
        for row in headline_rows
    }
    for offset, (family, variable) in enumerate(comparable):
        off_key, on_key = ("off", family, variable), ("on", family, variable)
        off_truth, off_prediction = test_groups[off_key]
        on_truth, on_prediction = test_groups[on_key]
        low, high = paired_trajectory_bootstrap_difference_ci(
            off_truth,
            off_prediction,
            on_truth,
            on_prediction,
            metric="r2",
            repeats=bootstrap_repeats,
            seed=seed + offset,
        )
        off_row, on_row = lookup[off_key], lookup[on_key]
        delta_rows.append({
            "model_seed": model_seed,
            "family": family,
            "variable": variable,
            "off_representation": off_row["representation"],
            "off_mode": off_row["mode"],
            "on_representation": on_row["representation"],
            "on_mode": on_row["mode"],
            "off_r2": off_row["r2"],
            "off_r2_ci_low": off_row["r2_ci_low"],
            "off_r2_ci_high": off_row["r2_ci_high"],
            "on_r2": on_row["r2"],
            "on_r2_ci_low": on_row["r2_ci_low"],
            "on_r2_ci_high": on_row["r2_ci_high"],
            "on_minus_off_r2": on_row["r2"] - off_row["r2"],
            "on_minus_off_r2_ci_low": low,
            "on_minus_off_r2_ci_high": high,
            "bootstrap_unit": "paired_complete_trajectory_window",
            "interval": "95_percentile",
        })
    return validation_rows, headline_rows, delta_rows


def readability_onset(rows, value_key, control_key, consecutive=2, fraction_of_peak=0.5):
    """Find the first layer above control and a fraction of the family peak."""
    ordered = sorted(rows, key=lambda row: int(row["layer"]))
    peak = max(float(row[value_key]) for row in ordered)
    qualifies = [
        float(row[value_key]) > max(float(row[control_key]), 0.0)
        and float(row[value_key]) >= fraction_of_peak * peak
        for row in ordered
    ]
    for index in range(0, len(ordered) - consecutive + 1):
        if all(qualifies[index : index + consecutive]):
            return int(ordered[index]["layer"])
    return None


def save_activation_cache(path, representations, states, actions, choices, metadata=None):
    """Save pooled activation arrays with a name map in one portable NPZ file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name_map = {f"rep_{index:03d}": name for index, name in enumerate(sorted(representations))}
    payload = {
        "states": np.asarray(states),
        "actions": np.asarray(actions),
        "choices": np.asarray(choices, dtype=np.int64),
        "metadata_json": np.asarray(json.dumps(metadata or {})),
        "name_map_json": np.asarray(json.dumps(name_map)),
    }
    for key, name in name_map.items():
        payload[key] = np.asarray(representations[name])
    np.savez_compressed(path, **payload)


def load_activation_cache(path):
    with np.load(Path(path), allow_pickle=False) as payload:
        name_map = json.loads(str(payload["name_map_json"].item()))
        representations = {name: payload[key] for key, name in name_map.items()}
        metadata = json.loads(str(payload["metadata_json"].item()))
        return (
            representations,
            payload["states"],
            payload["actions"],
            payload["choices"],
            metadata,
        )

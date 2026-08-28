import numpy as np

from scripts.umaze_probe_walkthrough import (
    align_representation,
    build_motion_targets,
    episode_group_split,
    episode_group_train_val_test_split,
    fit_probe_grouped,
    grouped_regression_summary,
    select_then_test_representations,
)


def test_temporal_alignment_shapes():
    states = np.zeros((8, 4, 4), dtype=np.float64)
    states[..., 0] = np.arange(4)[None, :]
    states[..., 2] = 1.0
    representations = np.arange(8 * 4 * 6, dtype=np.float64).reshape(8, 4, 6)
    targets = build_motion_targets(states, frameskip=1)

    x_position, y_position, _ = align_representation(
        representations, targets, "position", "frame"
    )
    x_velocity, y_velocity, _ = align_representation(
        representations, targets, "velocity", "delta"
    )
    x_acceleration, y_acceleration, _ = align_representation(
        representations, targets, "acceleration", "second_delta"
    )

    assert x_position.shape == (8, 4, 6)
    assert y_position.shape == (8, 4, 2)
    assert x_velocity.shape == (8, 3, 6)
    assert y_velocity.shape == (8, 3, 2)
    assert x_acceleration.shape == (8, 2, 6)
    assert y_acceleration.shape == (8, 2, 2)


def test_episode_split_does_not_leak_groups():
    choices = np.asarray([(episode, 0) for episode in range(20)], dtype=np.int64)
    train, test = episode_group_split(choices, test_fraction=0.2, seed=0)
    assert set(choices[train, 0]).isdisjoint(set(choices[test, 0]))
    assert len(train) == 16
    assert len(test) == 4


def test_three_way_episode_split_is_disjoint_and_complete():
    choices = np.asarray(
        [[episode, start] for episode in range(20) for start in (0, 4)],
        dtype=np.int64,
    )
    train, validation, test = episode_group_train_val_test_split(
        choices, validation_fraction=0.2, test_fraction=0.2, seed=3
    )
    groups = [set(choices[index, 0]) for index in (train, validation, test)]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert set.union(*groups) == set(range(20))


def test_grouped_probe_bootstraps_complete_windows():
    rng = np.random.default_rng(4)
    features = rng.normal(size=(30, 4, 3))
    labels = features[..., :2] @ np.asarray([[1.0, -0.4], [0.3, 0.8]])
    train = np.arange(20)
    test = np.arange(20, 30)
    truth, prediction, _ = fit_probe_grouped(features, labels, train, test, ridge=1e-4)
    summary = grouped_regression_summary(truth, prediction, repeats=50, seed=5)
    assert len(truth) == len(test)
    assert summary["test_trajectories"] == len(test)
    assert summary["r2"] > 0.99
    assert summary["r2_ci_low"] <= summary["r2"] <= summary["r2_ci_high"]


def test_validation_selects_signal_before_locked_test():
    rng = np.random.default_rng(6)
    target = rng.normal(size=(60, 3, 1))
    validation_winner = rng.normal(size=(60, 3, 2))
    validation_winner[:48, :, :1] = target[:48]
    test_winner = rng.normal(size=(60, 3, 2))
    test_winner[:36, :, :1] = target[:36]
    test_winner[48:, :, :1] = target[48:]
    representations = {
        condition: {
            "dino/0/validation_winner": validation_winner,
            "dino/1/test_winner": test_winner,
        }
        for condition in ("off", "on")
    }

    def align(rep, targets, variable, mode):
        return rep, targets[variable], np.zeros((60, 3, 2))

    validation, headline, deltas = select_then_test_representations(
        representations,
        {"position": target},
        align,
        lambda family, rep: [("position", "frame")],
        np.arange(36),
        np.arange(36, 48),
        np.arange(48, 60),
        ridge=1e-4,
        bootstrap_repeats=30,
        seed=7,
    )
    assert validation
    assert all(
        row["representation"] == "dino/0/validation_winner"
        for row in headline
    )
    assert all(row["r2"] < 0.5 for row in headline)
    assert all("r2_ci_low" in row and "mae_ci_high" in row for row in headline)
    assert len(deltas) == 1

import numpy as np

from scripts.umaze_probe_walkthrough import (
    align_representation,
    build_motion_targets,
    episode_group_split,
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

#!/usr/bin/env python3
"""Shared layer-wise activation extraction for the probing notebooks."""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from datasets.img_transforms import default_transform
from datasets.point_maze_dset import PointMazeDataset
from scripts.checkpoint_utils import resolve_checkpoint


def load_checkpoint(path: Path, device: torch.device):
    # Register torch-hub DINO classes before unpickling old checkpoints.
    from models.dino import DinoV2Encoder

    _ = DinoV2Encoder("dinov2_vits14", "x_norm_patchtokens")
    try:
        payload = torch.load(resolve_checkpoint(path), map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(resolve_checkpoint(path), map_location="cpu")
    modules = {}
    for name in ("encoder", "predictor", "proprio_encoder", "action_encoder"):
        if name not in payload:
            raise KeyError(f"checkpoint is missing {name!r}")
        modules[name] = payload[name].to(device).eval()
    return modules


def sample_windows(dataset: PointMazeDataset, count: int, frameskip: int, nframes: int, seed: int):
    rng = random.Random(seed)
    choices = []
    for episode, length in enumerate(dataset.seq_lengths.tolist()):
        max_start = int(length) - 1 - frameskip * (nframes - 1)
        if max_start >= 0:
            choices.extend((episode, start) for start in range(max_start + 1))
    rng.shuffle(choices)
    return choices[: min(count, len(choices))]


def load_batch(dataset, choices, frameskip, nframes):
    visuals, proprios, actions, states = [], [], [], []
    for episode, start in choices:
        indices = [start + frameskip * offset for offset in range(nframes)]
        obs, act, state, _ = dataset.get_frames(episode, indices)
        visuals.append(obs["visual"])
        proprios.append(obs["proprio"])
        actions.append(act)
        states.append(state)
    return (
        torch.stack(visuals),
        torch.stack(proprios),
        torch.stack(actions),
        torch.stack(states),
    )


def append(store, key, value):
    store.setdefault(key, []).append(value.detach().float().cpu())


def encode_stream(module, values, name):
    """Encode a stream, explicitly adapting legacy checkpoint input mismatch."""
    expected = int(module.patch_embed.in_channels)
    actual = int(values.shape[-1])
    if expected != actual:
        if actual > expected:
            raise ValueError(f"{name} has {actual} channels but checkpoint expects {expected}")
        print(
            f"WARNING: legacy {name} encoder expects {expected} channels but data has "
            f"{actual}; zero-padding for predictor activation probes",
            flush=True,
        )
        values = F.pad(values, (0, expected - actual))
    return module(values)


def collect_activations(
    modules,
    dataset,
    choices,
    batch_size,
    frameskip,
    nframes,
    device,
    include_kinds=None,
):
    """Collect intermediate representations for a set of trajectory windows.

    ``include_kinds`` optionally limits collection to representation suffixes
    such as ``{"cls", "pooled_patches", "projected_aggregate",
    "pooled_visual"}``.  The default keeps the original all-representations
    behavior.  The filter is useful for walkthroughs because retaining every
    individual patch at every layer can require several gigabytes.
    """
    encoder = modules["encoder"]
    predictor = modules["predictor"]
    representations = {}
    all_states, all_actions = [], []
    visual_dim = int(encoder.emb_dim)
    if hasattr(encoder, "agg_mlp"):
        token_count = int(encoder.agg_mlp[0].in_features // visual_dim)
        token_side = int(round(math.sqrt(token_count)))
        encoder_input_size = token_side * int(encoder.patch_size)
    else:
        encoder_input_size = 224

    def requested(kind):
        return include_kinds is None or kind in include_kinds

    with torch.inference_mode():
        for start in range(0, len(choices), batch_size):
            batch_choices = choices[start : start + batch_size]
            visual, proprio, action, state = load_batch(
                dataset, batch_choices, frameskip, nframes
            )
            b, t = visual.shape[:2]
            visual = visual.to(device)
            flat = visual.reshape(b * t, *visual.shape[2:])
            if flat.shape[-2:] != (encoder_input_size, encoder_input_size):
                flat = F.interpolate(
                    flat,
                    size=(encoder_input_size, encoder_input_size),
                    mode="bilinear",
                    align_corners=False,
                )
            layer_outputs = encoder.forward_intermediates(flat)
            for output in layer_outputs:
                layer = output["layer"]
                if requested("cls"):
                    append(representations, f"dino/{layer}/cls", output["cls"].reshape(b, t, -1))
                if requested("pooled_patches"):
                    append(
                        representations,
                        f"dino/{layer}/pooled_patches",
                        output["pooled_patches"].reshape(b, t, -1),
                    )
                if requested("individual_patches"):
                    append(
                        representations,
                        f"dino/{layer}/individual_patches",
                        output["patches"].reshape(b, t, output["patches"].shape[1], -1),
                    )
                if "projected" in output:
                    projected = output["projected"]
                    if projected.ndim == 2:
                        projected = projected.unsqueeze(1)
                    if requested("projected_patches"):
                        append(
                            representations,
                            f"dino/{layer}/projected_patches",
                            projected.reshape(b, t, projected.shape[1], -1),
                        )
                    if requested("projected_aggregate"):
                        append(
                            representations,
                            f"dino/{layer}/projected_aggregate",
                            output["aggregated"].reshape(b, t, -1),
                        )

            # Predictor activations use the exact final encoder representation
            # and normalized action/proprio streams used during training.
            visual_tokens = encoder(flat).reshape(b, t, -1, visual_dim)
            prop_emb = encode_stream(
                modules["proprio_encoder"], proprio.to(device), "proprio"
            )
            act_emb = encode_stream(
                modules["action_encoder"], action.to(device), "action"
            )
            prop_tiled = prop_emb.unsqueeze(2).expand(-1, -1, visual_tokens.shape[2], -1)
            act_tiled = act_emb.unsqueeze(2).expand(-1, -1, visual_tokens.shape[2], -1)
            z = torch.cat([visual_tokens, prop_tiled, act_tiled], dim=-1)
            hist = min(int(predictor.pos_embedding.shape[1] // z.shape[2]), t - 1)
            pred_input = z[:, :hist].reshape(b, hist * z.shape[2], -1)
            _, pred_layers = predictor(pred_input, return_intermediates=True)
            for layer, activation in enumerate(pred_layers):
                activation = activation.reshape(b, hist, z.shape[2], -1)[..., :visual_dim]
                if requested("pooled_visual"):
                    append(
                        representations,
                        f"predictor/{layer}/pooled_visual",
                        activation.mean(dim=2),
                    )
                if requested("individual_visual_tokens"):
                    append(
                        representations,
                        f"predictor/{layer}/individual_visual_tokens",
                        activation,
                    )

            all_states.append(state.float())
            all_actions.append(action.float())

    return (
        {key: torch.cat(value).numpy() for key, value in representations.items()},
        torch.cat(all_states).numpy(),
        torch.cat(all_actions).numpy(),
    )

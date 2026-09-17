"""
fedcare/attacks/model_poison.py - Model / Gradient Poisoning Attack Implementation.

Simulates adversarial clients that inject corrupted parameter updates into the
federation pipeline. Supports:
1. Sign-Flipping / Gradient Reversal: Scales update (w_local - w_global) by negative factor (e.g. -3.0).
2. Scaling Attack: Artificially scales update by a large positive multiplier (e.g. +10.0) to dominate aggregation.
3. Gaussian Noise Injection: Injects large-magnitude random perturbations.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np


def poison_weights(
    local_weights: Sequence[np.ndarray],
    global_weights: Sequence[np.ndarray] | None = None,
    mode: str = "sign_flip",
    scale: float = -3.0,
    noise_std: float = 0.5,
    seed: int = 42,
) -> list[np.ndarray]:
    """
    Poison local model weights prior to transmission to the central server.

    Parameters
    ----------
    local_weights : Sequence[np.ndarray]
        Parameters trained by the local client.
    global_weights : Sequence[np.ndarray] | None, optional
        Base global parameters received at the beginning of the round.
        Required for delta-based attacks (sign_flip, scaled_delta).
    mode : str, default="sign_flip"
        Attack type: 'sign_flip', 'scale', 'gaussian_noise', or 'zero'.
    scale : float, default=-3.0
        Multiplier applied to the update delta in delta-based attacks.
    noise_std : float, default=0.5
        Standard deviation of Gaussian noise in 'gaussian_noise' mode.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    list[np.ndarray]
        Adversarially altered parameter arrays matching the original model architecture.
    """
    rng = np.random.default_rng(seed)
    poisoned: list[np.ndarray] = []

    if mode == "sign_flip":
        if global_weights is None:
            # Fallback: invert local weights directly
            for w in local_weights:
                poisoned.append((w * scale).astype(w.dtype))
        else:
            for w_loc, w_glob in zip(local_weights, global_weights):
                delta = w_loc - w_glob
                # Steer global model strongly in opposite direction: w_glob + scale * delta
                w_att = w_glob + (scale * delta)
                poisoned.append(w_att.astype(w_loc.dtype))

    elif mode == "scale":
        if global_weights is None:
            for w in local_weights:
                poisoned.append((w * scale).astype(w.dtype))
        else:
            for w_loc, w_glob in zip(local_weights, global_weights):
                delta = w_loc - w_glob
                w_att = w_glob + (scale * delta)
                poisoned.append(w_att.astype(w_loc.dtype))

    elif mode == "gaussian_noise":
        for w in local_weights:
            noise = rng.normal(loc=0.0, scale=noise_std, size=w.shape).astype(w.dtype)
            poisoned.append(w + noise)

    elif mode == "zero":
        for w in local_weights:
            poisoned.append(np.zeros_like(w))

    else:
        raise ValueError(f"Unknown poisoning mode: {mode}")

    return poisoned

"""
fedcare/privacy.py - Differential Privacy Engine & Privacy Accountant for FedCare.

Implements:
1. Local / Client-Level Differential Privacy (LDP):
   - L2 Norm Clipping: Bounds the sensitivity of local model updates to ||Delta w||_2 <= C.
   - Calibrated Gaussian Mechanism: Injects noise N(0, (sigma * C)^2) to updates before aggregation.
2. Privacy Accounting:
   - Computes (epsilon, delta) privacy guarantees using the analytical Gaussian mechanism
     composition over communication rounds T.
   - Provides privacy-utility trade-off curves for clinical publication.
"""

from __future__ import annotations

import math
from typing import Sequence
import numpy as np


def clip_and_add_noise(
    local_weights: Sequence[np.ndarray],
    global_weights: Sequence[np.ndarray] | None = None,
    clip_norm: float = 1.0,
    noise_multiplier: float = 0.01,
    seed: int | None = None,
) -> list[np.ndarray]:
    """
    Apply L2 norm clipping and Gaussian noise to parameter updates.

    Parameters
    ----------
    local_weights : Sequence[np.ndarray]
        Trained local model weights.
    global_weights : Sequence[np.ndarray] | None, optional
        Starting global model weights. If provided, clipping and noise are applied
        to the update delta (w_local - w_global).
    clip_norm : float, default=1.0
        Maximum allowable L2 norm threshold C.
    noise_multiplier : float, default=0.01
        Noise scale relative to clip_norm: sigma = noise_multiplier * clip_norm.
        If 0.0, no noise is added.
    seed : int | None, optional
        Seed for reproducibility.

    Returns
    -------
    list[np.ndarray]
        Differentially private parameter weights.
    """
    rng = np.random.default_rng(seed)

    if global_weights is not None:
        deltas = [w_loc - w_glob for w_loc, w_glob in zip(local_weights, global_weights)]
    else:
        deltas = [w_loc.copy() for w_loc in local_weights]

    # Calculate global L2 norm of the update delta across all layers
    total_norm_sq = sum(float(np.sum(d ** 2)) for d in deltas)
    total_norm = math.sqrt(total_norm_sq)

    # 1. L2 Norm Clipping: scale by min(1, C / ||delta||)
    clip_factor = 1.0
    if clip_norm > 0 and total_norm > clip_norm:
        clip_factor = clip_norm / total_norm

    clipped_deltas = [d * clip_factor for d in deltas]

    # 2. Gaussian Noise Injection: N(0, (sigma * C)^2)
    private_weights: list[np.ndarray] = []
    sigma = noise_multiplier * clip_norm if clip_norm > 0 else noise_multiplier

    for i, d in enumerate(clipped_deltas):
        if sigma > 0.0:
            noise = rng.normal(loc=0.0, scale=sigma, size=d.shape).astype(d.dtype)
            d_noisy = d + noise
        else:
            d_noisy = d

        if global_weights is not None:
            w_priv = global_weights[i] + d_noisy
        else:
            w_priv = d_noisy

        private_weights.append(w_priv.astype(local_weights[i].dtype))

    return private_weights


class PrivacyAccountant:
    """
    Rényi / Advanced Composition Privacy Accountant for the Gaussian Mechanism.

    Computes epsilon given noise multiplier sigma, number of rounds T,
    and target delta.
    """

    @staticmethod
    def compute_epsilon(
        noise_multiplier: float,
        num_rounds: int,
        delta: float = 1e-5,
    ) -> float:
        """
        Compute total privacy loss epsilon for the Gaussian mechanism over T rounds.

        Using standard advanced composition:
            epsilon = (sqrt(2 * ln(1.25 / delta)) * sqrt(T)) / noise_multiplier

        Parameters
        ----------
        noise_multiplier : float
            Noise parameter sigma.
        num_rounds : int
            Total communication rounds T.
        delta : float, default=1e-5
            Probability of privacy breach (usually delta < 1 / N_patients).

        Returns
        -------
        float
            Privacy parameter epsilon. Lower epsilon indicates stronger privacy.
            Returns math.inf if noise_multiplier == 0.0.
        """
        if noise_multiplier <= 0.0:
            return float("inf")

        if num_rounds <= 0:
            return 0.0

        if delta <= 0.0 or delta >= 1.0:
            raise ValueError(f"Delta must be in (0, 1), got {delta}")

        # Standard analytical Gaussian composition bound
        # epsilon = sqrt(2 * T * ln(1.25 / delta)) / sigma
        numerator = math.sqrt(2.0 * num_rounds * math.log(1.25 / delta))
        epsilon = numerator / noise_multiplier
        return round(float(epsilon), 3)

    @staticmethod
    def compute_noise_multiplier(
        epsilon: float,
        num_rounds: int,
        delta: float = 1e-5,
    ) -> float:
        """
        Compute required noise multiplier for a target epsilon over T rounds.
        """
        if epsilon <= 0.0:
            return float("inf")
        if num_rounds <= 0:
            return 0.0
        numerator = math.sqrt(2.0 * num_rounds * math.log(1.25 / delta))
        return float(numerator / epsilon)

    @staticmethod
    def get_privacy_regime_description(epsilon: float) -> str:
        """Provide a human-readable interpretation of the privacy level."""
        if math.isinf(epsilon) or epsilon >= 10.0:
            return "No Privacy Guarantee (Functional)"
        elif epsilon < 1.0:
            return f"Very Strong Privacy (eps = {epsilon:.2f})"
        elif epsilon < 5.0:
            return f"Strong Privacy (eps = {epsilon:.2f})"
        else:
            return f"Moderate Privacy (eps = {epsilon:.2f})"

"""
tests/test_phase4.py - Unit & Integration Tests for FedCare Phase 4.

Covers:
1. Adversarial Attacks: Label-Flipping and Model Poisoning (Sign-Flip, Scaling, Noise).
2. Byzantine Defenses: Trimmed Mean, Coordinate-wise Median, and Multi-Krum.
3. Differential Privacy: L2 Norm Clipping, Gaussian Noise, and Privacy Accounting.
4. Communication Cost: Parameter payload sizing and federation efficiency accounting.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fedcare.attacks.label_flip import apply_label_flip, poison_dataloader_labels
from fedcare.attacks.model_poison import poison_weights
from fedcare.comm_cost import compute_federation_comm_cost, compute_parameter_size
from fedcare.privacy import PrivacyAccountant, clip_and_add_noise
from fedcare.strategy.krum import aggregate_krum
from fedcare.strategy.median import aggregate_median
from fedcare.strategy.trimmed_mean import aggregate_trimmed_mean


class TestAttacks:
    """Verify adversarial attack mechanics."""

    def test_label_flip_dataframe(self) -> None:
        df = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0, 4.0],
            "target": [0, 1, 0, 1],
        })
        # 100% flip
        flipped = apply_label_flip(df, flip_rate=1.0, seed=42)
        assert (flipped["target"].to_numpy() == np.array([1, 0, 1, 0])).all()

        # 0% flip
        unflipped = apply_label_flip(df, flip_rate=0.0, seed=42)
        assert (unflipped["target"].to_numpy() == np.array([0, 1, 0, 1])).all()

    def test_label_flip_dataloader(self) -> None:
        x = torch.randn(10, 5)
        y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)

        poisoned_loader = poison_dataloader_labels(loader, flip_rate=1.0, seed=42)
        all_y_poisoned = []
        for _, batch_y in poisoned_loader:
            all_y_poisoned.append(batch_y)
        y_out = torch.cat(all_y_poisoned)

        assert len(y_out) == 10
        # Check that inverted labels are mostly 1 for original 0 and 0 for original 1
        assert (y_out == 1 - y).sum().item() == 10

    def test_model_poison_sign_flip(self) -> None:
        global_w = [np.zeros((3, 3), dtype=np.float32)]
        local_w = [np.ones((3, 3), dtype=np.float32) * 2.0]  # delta = +2.0

        # Scale by -3.0 -> update should be 0 + (-3.0) * 2.0 = -6.0
        poisoned = poison_weights(
            local_w, global_weights=global_w, mode="sign_flip", scale=-3.0
        )
        assert np.allclose(poisoned[0], -6.0)

    def test_model_poison_scaling(self) -> None:
        global_w = [np.ones((2, 2), dtype=np.float32)]
        local_w = [np.ones((2, 2), dtype=np.float32) * 1.5]  # delta = +0.5

        # Scale by 10.0 -> update should be 1.0 + 10.0 * 0.5 = 6.0
        poisoned = poison_weights(
            local_w, global_weights=global_w, mode="scale", scale=10.0
        )
        assert np.allclose(poisoned[0], 6.0)

    def test_model_poison_gaussian_noise(self) -> None:
        local_w = [np.zeros((10, 10), dtype=np.float32)]
        poisoned = poison_weights(local_w, mode="gaussian_noise", noise_std=1.0, seed=42)
        assert not np.allclose(poisoned[0], 0.0)
        assert abs(np.mean(poisoned[0])) < 0.3
        assert 0.7 < np.std(poisoned[0]) < 1.3


class TestDefenses:
    """Verify Byzantine-tolerant aggregation strategies."""

    def test_aggregate_trimmed_mean(self) -> None:
        # 5 clients: 4 normal updates around 1.0, 1 extreme poisoned update at 100.0
        results = [
            ([np.array([1.0, 1.0], dtype=np.float32)], 100),
            ([np.array([1.1, 0.9], dtype=np.float32)], 100),
            ([np.array([0.9, 1.1], dtype=np.float32)], 100),
            ([np.array([1.0, 1.0], dtype=np.float32)], 100),
            ([np.array([100.0, 100.0], dtype=np.float32)], 100),  # Poisoned
        ]
        # Trim 1 from each end (beta = 0.2)
        agg = aggregate_trimmed_mean(results, beta=0.2)
        # Middle 3 values for coord 0: [0.9, 1.0, 1.0, 1.1, 100.0] -> trimmed: [1.0, 1.0, 1.1] -> mean ~ 1.033
        assert agg[0][0] < 2.0
        assert np.allclose(agg[0], [1.0, 1.0], atol=0.1)

    def test_aggregate_median(self) -> None:
        # 5 clients with an extreme outlier
        results = [
            ([np.array([2.0, 4.0], dtype=np.float32)], 100),
            ([np.array([2.1, 3.9], dtype=np.float32)], 100),
            ([np.array([1.9, 4.1], dtype=np.float32)], 100),
            ([np.array([2.0, 4.0], dtype=np.float32)], 100),
            ([np.array([-500.0, 999.0], dtype=np.float32)], 100),  # Poisoned
        ]
        agg = aggregate_median(results)
        assert np.allclose(agg[0], [2.0, 4.0], atol=0.15)

    def test_aggregate_krum(self) -> None:
        # 5 clients: 4 clustered near [1.0, 1.0], 1 far away at [50.0, 50.0]
        results = [
            ([np.array([1.0, 1.0], dtype=np.float32)], 100),
            ([np.array([1.05, 0.95], dtype=np.float32)], 100),
            ([np.array([0.95, 1.05], dtype=np.float32)], 100),
            ([np.array([1.0, 1.0], dtype=np.float32)], 100),
            ([np.array([50.0, 50.0], dtype=np.float32)], 100),  # Poisoned Byzantine
        ]
        # Krum should reject client 5 and average the inliers
        agg = aggregate_krum(results, num_malicious=1, num_to_keep=3)
        assert np.all(agg[0] < 2.0)
        assert np.allclose(agg[0], [1.0, 1.0], atol=0.1)


class TestDifferentialPrivacy:
    """Verify Differential Privacy engine and accountant."""

    def test_clip_and_add_noise(self) -> None:
        global_w = [np.zeros((4, 4), dtype=np.float32)]
        local_w = [np.ones((4, 4), dtype=np.float32) * 5.0]  # norm = sqrt(16 * 25) = 20.0

        # Clip at C = 2.0 with zero noise
        clipped = clip_and_add_noise(
            local_w, global_weights=global_w, clip_norm=2.0, noise_multiplier=0.0
        )
        clipped_norm = math.sqrt(float(np.sum(clipped[0] ** 2)))
        assert np.isclose(clipped_norm, 2.0, atol=1e-4)

        # Adding Gaussian noise
        noisy = clip_and_add_noise(
            local_w, global_weights=global_w, clip_norm=2.0, noise_multiplier=0.1, seed=42
        )
        assert not np.allclose(noisy[0], clipped[0])

    def test_privacy_accountant_epsilon(self) -> None:
        # With zero noise, epsilon is infinite
        assert math.isinf(PrivacyAccountant.compute_epsilon(0.0, num_rounds=10))

        # Standard parameters
        eps1 = PrivacyAccountant.compute_epsilon(noise_multiplier=0.1, num_rounds=15, delta=1e-5)
        eps2 = PrivacyAccountant.compute_epsilon(noise_multiplier=0.5, num_rounds=15, delta=1e-5)

        # Higher noise multiplier -> stronger privacy (lower epsilon)
        assert eps2 < eps1
        assert eps1 > 0.0


class TestCommunicationCost:
    """Verify communication cost tracking."""

    def test_compute_parameter_size(self) -> None:
        w1 = np.zeros((10, 10), dtype=np.float32)  # 100 floats = 400 bytes
        w2 = np.zeros((5,), dtype=np.float32)      # 5 floats = 20 bytes
        info = compute_parameter_size([w1, w2])

        assert info["num_parameters"] == 105.0
        assert info["bytes"] == 420.0
        assert info["kb"] > 0.4

    def test_compute_federation_comm_cost(self) -> None:
        w = [np.zeros((3042,), dtype=np.float32)]  # FedCare MLP size
        cost = compute_federation_comm_cost(
            num_clients=6,
            num_rounds=20,
            parameters=w,
            centralized_dataset_bytes=490_000,
        )
        assert cost["param_count"] == 3042.0
        assert cost["single_message_kb"] > 11.0
        assert cost["total_fl_mb"] > 0.0
        assert "comm_ratio" in cost

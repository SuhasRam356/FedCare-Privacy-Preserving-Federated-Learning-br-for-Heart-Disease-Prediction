"""
tests/test_phase3.py - Comprehensive test suite for Phase 3 components.

Tests:
1. IID and Dirichlet Non-IID partitioning logic in fedcare/partition.py.
2. FedProx proximal regularization in task.py and client_app.py.
3. Client-side personalization / local fine-tuning.
4. Federated fairness metrics in fedcare/metrics.py.
"""

import numpy as np
import pytest
import torch

from fedcare.client_app import FlowerClient, get_parameters, set_parameters
from fedcare.metrics import (
    compute_classification_metrics,
    compute_fairness_metrics,
)
from fedcare.partition import (
    get_partition_data,
    partition_dirichlet,
    partition_iid,
)
from fedcare.strategy.fedprox import FedProx
from fedcare.task import Net, train


class TestPartitioning:
    """Tests for dataset partitioning modules."""

    def test_iid_partitioning_counts(self) -> None:
        """IID partitioning should create 6 partitions with equal sample counts."""
        partitions = partition_iid(num_partitions=6, batch_size=32, seed=42)
        assert len(partitions) == 6

        # Check all partitions have roughly equal train sizes (~1600)
        sizes = [len(p[0].dataset) for p in partitions.values()]
        assert all(s == 1600 for s in sizes)

    def test_dirichlet_partitioning(self) -> None:
        """Dirichlet partitioning should create 6 partitions across classes."""
        partitions = partition_dirichlet(alpha=0.5, num_partitions=6, batch_size=32, seed=42)
        assert len(partitions) == 6
        for train_loader, test_loader, scaler in partitions.values():
            assert len(train_loader.dataset) > 0
            assert len(test_loader.dataset) > 0

    def test_get_partition_data_unified_api(self) -> None:
        """Unified partition loader should handle all 3 regimes."""
        tr1, te1, _ = get_partition_data("hospital_native", partition_id=1)
        assert len(tr1.dataset) == 1600

        tr2, te2, _ = get_partition_data("iid", partition_id=2)
        assert len(tr2.dataset) == 1600

        tr3, te3, _ = get_partition_data("dirichlet", partition_id=3, alpha=0.5)
        assert len(tr3.dataset) > 0


class TestFedProxMechanics:
    """Tests for FedProx proximal loss and client execution."""

    def test_proximal_loss_computation(self) -> None:
        """Training with proximal term should complete and return non-negative loss."""
        model = Net()
        global_model = Net()
        # Create dummy dataloader
        from torch.utils.data import DataLoader, TensorDataset
        X = torch.randn(64, 13)
        y = torch.randint(0, 2, (64,))
        loader = DataLoader(TensorDataset(X, y), batch_size=32)

        loss = train(
            model=model,
            train_loader=loader,
            epochs=1,
            lr=0.01,
            mu=0.1,
            global_model=global_model,
        )
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_fedprox_client_fit_with_mu(self) -> None:
        """FlowerClient should accept mu parameter and train with proximal regularization."""
        client = FlowerClient(partition_id=1, batch_size=64)
        params = client.get_parameters(config={})

        updated_params, num_samples, metrics = client.fit(
            parameters=params,
            config={"local_epochs": 1, "lr": 0.001, "mu": 0.1},
        )
        assert len(updated_params) == len(params)
        assert metrics["mu"] == 0.1
        assert "train_loss" in metrics

    def test_client_personalization(self) -> None:
        """Client personalization should perform local fine-tuning."""
        client = FlowerClient(partition_id=2, batch_size=64)
        metrics = client.personalize(epochs=1, lr=0.001)

        assert "loss" in metrics
        assert "accuracy" in metrics
        assert "auc" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0


class TestFairnessAndMetrics:
    """Tests for classification and federated fairness accounting."""

    def test_compute_classification_metrics(self) -> None:
        """Standard metrics calculation."""
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_probs = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.4])
        metrics = compute_classification_metrics(y_true, y_probs)

        assert metrics["accuracy"] == 1.0
        assert metrics["auc"] == 1.0
        assert metrics["f1"] == 1.0

    def test_compute_fairness_metrics(self) -> None:
        """Fairness accounting should compute worst, best, and equity gap correctly."""
        client_metrics = {
            1: {"auc": 0.88, "accuracy": 0.85},
            2: {"auc": 0.78, "accuracy": 0.95},
            3: {"auc": 0.84, "accuracy": 0.80},
        }
        fairness = compute_fairness_metrics(client_metrics, metric_key="auc")

        assert fairness["auc_worst"] == 0.78
        assert fairness["auc_best"] == 0.88
        assert pytest.approx(fairness["auc_equity_gap"], 1e-4) == 0.10

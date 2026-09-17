"""
tests/test_federated.py - Comprehensive test suite for Phase 2 federated components.

Verifies:
1. PyTorch state_dict <-> Flower NDArrays parameter serialization.
2. FlowerClient instantiation, fit, and evaluate across hospital partitions.
3. Sample-weighted aggregation logic in FedAvgWeighted.
4. End-to-end mini federated learning loop execution.
"""

from pathlib import Path
import numpy as np
import pytest
import torch
from flwr.common import (
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from fedcare.client_app import (
    FlowerClient,
    client_fn,
    get_parameters,
    set_parameters,
)
from fedcare.server_app import get_evaluate_fn, get_initial_parameters
from fedcare.strategy.fedavg_weighted import (
    FedAvgWeighted,
    weighted_average_metrics,
)
from fedcare.task import Net


class TestParameterSerialization:
    """Tests for parameter serialization between PyTorch and Flower."""

    def test_get_and_set_parameters_roundtrip(self) -> None:
        """Parameters extracted and re-applied should perfectly match."""
        net1 = Net()
        net2 = Net()

        # Extract weights from net1
        params = get_parameters(net1)
        assert isinstance(params, list)
        assert len(params) > 0

        # Load weights into net2
        set_parameters(net2, params)

        # Check all tensors match exactly
        for (k1, v1), (k2, v2) in zip(net1.state_dict().items(), net2.state_dict().items()):
            assert k1 == k2
            assert torch.allclose(v1, v2, atol=1e-7)

    def test_parameter_shape_match(self) -> None:
        """Extracted array shapes must match net state_dict shapes."""
        net = Net()
        params = get_parameters(net)
        state_dict = net.state_dict()
        assert len(params) == len(state_dict)
        for arr, (_, tensor) in zip(params, state_dict.items()):
            assert arr.shape == tuple(tensor.shape)


class TestFlowerClient:
    """Tests for the hospital FlowerClient implementation."""

    def test_client_initialization(self) -> None:
        """Client should successfully initialize with partition data."""
        client = FlowerClient(partition_id=1, batch_size=32)
        assert client.partition_id == 1
        assert client.num_train_samples > 0
        assert client.num_test_samples > 0
        assert isinstance(client.net, Net)

    def test_client_invalid_partition(self) -> None:
        """FlowerClient and client_fn should reject invalid partition IDs."""
        with pytest.raises(ValueError):
            FlowerClient(partition_id=0)
        with pytest.raises(ValueError):
            FlowerClient(partition_id=7)
        with pytest.raises(ValueError):
            client_fn(-1)
        with pytest.raises(ValueError):
            client_fn(10)

    def test_client_get_parameters(self) -> None:
        """Client get_parameters should return list of numpy arrays."""
        client = FlowerClient(partition_id=2, batch_size=32)
        params = client.get_parameters(config={})
        assert isinstance(params, list)
        assert all(isinstance(p, np.ndarray) for p in params)

    def test_client_fit(self) -> None:
        """Client fit should update weights and return valid loss."""
        client = FlowerClient(partition_id=1, batch_size=64)
        initial_params = client.get_parameters(config={})

        updated_params, num_samples, metrics = client.fit(
            parameters=initial_params,
            config={"local_epochs": 1, "lr": 0.01},
        )
        assert len(updated_params) == len(initial_params)
        assert num_samples == client.num_train_samples
        assert "train_loss" in metrics
        assert isinstance(metrics["train_loss"], float)
        assert metrics["train_loss"] >= 0.0

    def test_client_evaluate(self) -> None:
        """Client evaluate should return loss, accuracy, and AUC."""
        client = FlowerClient(partition_id=3, batch_size=64)
        params = client.get_parameters(config={})

        loss, num_samples, metrics = client.evaluate(
            parameters=params,
            config={},
        )
        assert num_samples == client.num_test_samples
        assert isinstance(loss, float)
        assert "accuracy" in metrics
        assert "auc" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["auc"] <= 1.0


class TestStrategyWeighted:
    """Tests for FedAvgWeighted strategy and metric aggregation."""

    def test_weighted_average_metrics(self) -> None:
        """Metrics should be averaged proportionally to sample counts."""
        # Hospital 1: 100 samples, acc=0.80, auc=0.85
        # Hospital 2: 300 samples, acc=0.90, auc=0.95
        # Weighted acc = (100*0.80 + 300*0.90) / 400 = (80 + 270)/400 = 350/400 = 0.875
        # Weighted auc = (100*0.85 + 300*0.95) / 400 = (85 + 285)/400 = 370/400 = 0.925
        metrics_input = [
            (100, {"accuracy": 0.80, "auc": 0.85}),
            (300, {"accuracy": 0.90, "auc": 0.95}),
        ]
        result = weighted_average_metrics(metrics_input)
        assert pytest.approx(result["accuracy"], 1e-5) == 0.875
        assert pytest.approx(result["auc"], 1e-5) == 0.925

    def test_weighted_average_empty(self) -> None:
        """Empty metric list should return empty dict."""
        assert weighted_average_metrics([]) == {}

    def test_strategy_initialization(self) -> None:
        """Strategy should initialize with correct parameter configuration."""
        initial_params = get_initial_parameters()
        strategy = FedAvgWeighted(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=6,
            min_evaluate_clients=6,
            min_available_clients=6,
            initial_parameters=initial_params,
        )
        assert strategy.min_fit_clients == 6
        assert strategy.initial_parameters is not None


class TestMiniFederationLoop:
    """End-to-end integration test of a mini federated loop."""

    def test_two_round_simulation(self) -> None:
        """Run a 2-round federated training on 2 clients to verify aggregation flow."""
        clients = [FlowerClient(partition_id=1), FlowerClient(partition_id=2)]
        net = Net()
        global_params = get_parameters(net)

        for rnd in range(1, 3):
            fit_results = []
            for c in clients:
                weights, n_samples, metrics = c.fit(
                    parameters=global_params,
                    config={"local_epochs": 1, "lr": 0.005},
                )
                fit_results.append((weights, n_samples))

            # Aggregate
            total_n = sum(n for _, n in fit_results)
            new_global = [np.zeros_like(w) for w in global_params]
            for w, n in fit_results:
                fraction = n / total_n
                for i, layer in enumerate(w):
                    new_global[i] += layer * fraction

            global_params = new_global

        # Evaluate on client 1
        loss, samples, metrics = clients[0].evaluate(
            parameters=global_params,
            config={},
        )
        assert isinstance(loss, float)
        assert 0.0 <= metrics["auc"] <= 1.0

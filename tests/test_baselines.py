"""
tests/test_baselines.py – Smoke tests for the core FedCare components.

Verifies:
    1. The Net model can be instantiated and produces correct output shape.
    2. load_data returns working DataLoaders for both centralized and
       partitioned modes.
    3. train() and test() run without errors.
"""

import pytest
import torch

from fedcare.task import Net, HeartDiseaseDataset, load_data, train, evaluate, NUM_FEATURES, NUM_CLASSES


# ── Model tests ───────────────────────────────────────────────────────

class TestNet:
    """Tests for the MLP model."""

    def test_instantiation(self):
        """Net() should create without errors."""
        model = Net()
        assert model is not None

    def test_output_shape(self):
        """Forward pass should produce (batch_size, NUM_CLASSES) logits."""
        model = Net()
        batch = torch.randn(8, NUM_FEATURES)
        output = model(batch)
        assert output.shape == (8, NUM_CLASSES)

    def test_parameter_count(self):
        """Model should have a reasonable number of parameters."""
        model = Net()
        total_params = sum(p.numel() for p in model.parameters())
        # 13*64 + 64 + 64*32 + 32 + 32*2 + 2 = 832+64+2048+32+64+2 = 3042
        assert total_params == 3042


# ── Dataset tests ─────────────────────────────────────────────────────

class TestHeartDiseaseDataset:
    """Tests for the custom PyTorch Dataset."""

    def test_length(self):
        import numpy as np
        features = np.random.randn(50, NUM_FEATURES)
        labels = np.random.randint(0, 2, size=50)
        ds = HeartDiseaseDataset(features, labels)
        assert len(ds) == 50

    def test_getitem(self):
        import numpy as np
        features = np.random.randn(10, NUM_FEATURES)
        labels = np.random.randint(0, 2, size=10)
        ds = HeartDiseaseDataset(features, labels)
        x, y = ds[0]
        assert x.shape == (NUM_FEATURES,)
        assert y.shape == ()


# ── Data loading tests ────────────────────────────────────────────────

class TestLoadData:
    """Tests for the load_data function (requires generated CSVs)."""

    def test_centralized_loading(self):
        """load_data(None) should return train/test loaders from combined.csv."""
        train_loader, test_loader, scaler = load_data(partition_id=None)
        assert len(train_loader) > 0
        assert len(test_loader) > 0

    def test_partition_loading(self):
        """load_data(1) should return loaders from hospital_1.csv."""
        train_loader, test_loader, scaler = load_data(partition_id=1)
        assert len(train_loader) > 0
        assert len(test_loader) > 0

    def test_batch_shapes(self):
        """Batches should have correct feature dimensions."""
        train_loader, _, _ = load_data(partition_id=None, batch_size=16)
        features, labels = next(iter(train_loader))
        assert features.shape[1] == NUM_FEATURES
        assert labels.ndim == 1


# ── Train / test loop tests ───────────────────────────────────────────

class TestTrainAndTest:
    """Smoke tests for the training and evaluation loops."""

    def test_train_returns_loss(self):
        """train() should return a finite float loss."""
        train_loader, _, _ = load_data(partition_id=1)
        model = Net()
        loss = train(model, train_loader, epochs=1)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_test_returns_metrics(self):
        """evaluate() should return a dict with loss, accuracy, auc."""
        _, test_loader, _ = load_data(partition_id=1)
        model = Net()
        metrics = evaluate(model, test_loader)
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert "auc" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

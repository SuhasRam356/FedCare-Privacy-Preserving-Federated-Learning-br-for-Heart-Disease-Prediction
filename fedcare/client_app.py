"""
fedcare.client_app - Flower NumPyClient implementation for hospital nodes.

Each hospital represents an autonomous federated client participating in
the FedCare training protocol. Hospitals never exchange raw patient records.
Only model parameter tensors (weights and biases) are communicated.

Key features:
- PyTorch state_dict <-> NumPy parameter serialization
- Configurable local epochs and learning rate per round
- Local evaluation on held-out hospital validation split
- Full type annotations, comprehensive docstrings, and robust error handling
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from flwr.client import Client, NumPyClient
from flwr.common import (
    Context,
    NDArrays,
    Scalar,
)

from fedcare.task import (
    BATCH_SIZE,
    HeartDiseaseDataset,
    Net,
    evaluate,
    load_data,
    train,
)

logger = logging.getLogger("fedcare.client_app")


def get_parameters(net: nn.Module) -> NDArrays:
    """
    Extract neural network parameters as a list of NumPy ndarrays.

    Parameters
    ----------
    net : nn.Module
        PyTorch neural network.

    Returns
    -------
    NDArrays
        List of NumPy ndarrays corresponding to state_dict parameter tensors.
    """
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net: nn.Module, parameters: NDArrays) -> None:
    """
    Load parameters from a list of NumPy ndarrays into a PyTorch network.

    Parameters
    ----------
    net : nn.Module
        PyTorch neural network to update in-place.
    parameters : NDArrays
        List of parameter arrays from the federated server.

    Raises
    ------
    ValueError
        If the number of parameter arrays does not match network layers.
    """
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = {
        k: torch.tensor(v, dtype=net.state_dict()[k].dtype)
        for k, v in params_dict
    }
    net.load_state_dict(state_dict, strict=True)


class FlowerClient(NumPyClient):
    """
    Flower client representing a single hospital participating in FedCare.

    Attributes
    ----------
    partition_id : int
        Hospital identifier (1 to 6).
    net : Net
        Local instance of the FedCare MLP classifier.
    train_loader : DataLoader
        DataLoader for hospital's local training split.
    test_loader : DataLoader
        DataLoader for hospital's local test split.
    device : torch.device
        Computation device ('cuda' if available, else 'cpu').
    """

    def __init__(
        self,
        partition_id: int,
        batch_size: int = BATCH_SIZE,
        device: Optional[torch.device] = None,
        partition_type: str = "hospital_native",
        alpha: Optional[float] = None,
        seed: int = 42,
        attack_type: Optional[str] = None,
        flip_rate: float = 1.0,
        poison_mode: str = "sign_flip",
        poison_scale: float = -3.0,
        dp_clip_norm: float = 0.0,
        dp_noise_multiplier: float = 0.0,
    ) -> None:
        """
        Initialize the FlowerClient for a specific hospital partition.

        Parameters
        ----------
        partition_id : int
            Hospital index (1 through 6).
        batch_size : int, optional
            Mini-batch size for training/evaluation (default: 32).
        device : torch.device, optional
            Computation device (CPU/GPU).
        partition_type : str, optional
            Dataset distribution ('hospital_native', 'iid', 'dirichlet').
        alpha : float, optional
            Concentration parameter for Dirichlet partitioning.
        seed : int, optional
            Random seed for reproducibility.
        attack_type : str, optional
            Adversarial mode: None, 'label_flip', or 'model_poison'.
        flip_rate : float, optional
            Fraction of labels to invert in 'label_flip' mode.
        poison_mode : str, optional
            Model poisoning variant ('sign_flip', 'scale', 'gaussian_noise').
        poison_scale : float, optional
            Multiplier for model update poisoning.
        dp_clip_norm : float, optional
            L2 gradient clipping norm threshold (0.0 means disabled).
        dp_noise_multiplier : float, optional
            Gaussian noise multiplier for Differential Privacy.
        """
        super().__init__()
        if partition_id < 1 or partition_id > 6:
            raise ValueError(
                f"Invalid partition_id {partition_id}. Must be between 1 and 6."
            )

        self.partition_id = partition_id
        self.batch_size = batch_size
        self.partition_type = partition_type
        self.alpha = alpha
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.attack_type = attack_type
        self.flip_rate = flip_rate
        self.poison_mode = poison_mode
        self.poison_scale = poison_scale
        self.dp_clip_norm = dp_clip_norm
        self.dp_noise_multiplier = dp_noise_multiplier
        self.seed = seed

        self.net = Net().to(self.device)

        from fedcare.partition import get_partition_data

        self.train_loader, self.test_loader, self.scaler = get_partition_data(
            partition_type=partition_type,
            partition_id=partition_id,
            alpha=alpha,
            batch_size=batch_size,
            seed=seed,
        )

        # Apply label-flipping attack if active on this client
        if self.attack_type == "label_flip":
            from fedcare.attacks.label_flip import poison_dataloader_labels
            self.train_loader = poison_dataloader_labels(
                self.train_loader,
                flip_rate=self.flip_rate,
                seed=self.seed,
            )
            logger.warning(
                "Hospital %d: ACTIVE ADVERSARY -> Label-flip attack applied (rate=%.2f).",
                self.partition_id,
                self.flip_rate,
            )

        self.num_train_samples = len(self.train_loader.dataset)
        self.num_test_samples = len(self.test_loader.dataset)

        logger.debug(
            "Initialized FlowerClient for Hospital %d (%s, %d train, %d test samples).",
            self.partition_id,
            self.partition_type,
            self.num_train_samples,
            self.num_test_samples,
        )

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        """
        Return the current local model parameters.

        Parameters
        ----------
        config : dict[str, Scalar]
            Configuration dictionary passed by the server.

        Returns
        -------
        NDArrays
            Current local model weights and biases as NumPy ndarrays.
        """
        return get_parameters(self.net)

    def fit(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """
        Train local model on hospital data starting from global parameters.

        Supports FedProx proximal regularization if 'mu' > 0.0 in config,
        model/gradient poisoning attacks, and Differential Privacy (DP).
        """
        # 1. Update local model with global weights
        set_parameters(self.net, parameters)

        # [NEW] Server Defense: Verify the global model before training
        verify_server = config.get("verify_server", False)
        if verify_server:
            metrics_before = evaluate(self.net, self.test_loader, device=self.device)
            # A completely scrambled model usually gets ~0.50 (random guessing). 
            # If accuracy and AUC are both poor, it's a poisoned model.
            if metrics_before["accuracy"] < 0.55 and metrics_before["auc"] < 0.55:
                logger.error(f"Hospital {self.partition_id}: SECURITY ALERT! Received poisoned global model from server (Accuracy: {metrics_before['accuracy']:.2f}, AUC: {metrics_before['auc']:.2f}). Aborting training.")
                # Raise exception to abort the malicious update
                raise RuntimeError("Server Compromise Detected. Training Aborted.")

        # 2. Extract hyperparameters from config
        local_epochs = int(config.get("local_epochs", 2))
        learning_rate = float(config.get("lr", 0.001))
        mu = float(config.get("mu", 0.0))

        # Dynamic attack or DP overrides from server config if present
        attack_type = str(config.get("attack_type", self.attack_type or ""))
        if not attack_type:
            attack_type = self.attack_type or ""

        dp_clip_norm = float(config.get("dp_clip_norm", self.dp_clip_norm))
        dp_noise_multiplier = float(config.get("dp_noise_multiplier", self.dp_noise_multiplier))

        # 3. Setup global model snapshot for FedProx proximal term
        global_model = None
        if mu > 0.0:
            global_model = Net().to(self.device)
            set_parameters(global_model, parameters)
            for param in global_model.parameters():
                param.requires_grad = False

        # 4. Perform local gradient updates
        train_loss = train(
            self.net,
            self.train_loader,
            epochs=local_epochs,
            lr=learning_rate,
            device=self.device,
            mu=mu,
            global_model=global_model,
        )

        local_weights = get_parameters(self.net)

        # 5. Apply Model Poisoning Attack if configured
        if attack_type == "model_poison":
            from fedcare.attacks.model_poison import poison_weights
            local_weights = poison_weights(
                local_weights,
                global_weights=parameters,
                mode=self.poison_mode,
                scale=self.poison_scale,
                seed=self.seed,
            )

        # 6. Apply Differential Privacy (L2 Clipping + Gaussian Noise) if active
        if dp_clip_norm > 0.0 or dp_noise_multiplier > 0.0:
            from fedcare.privacy import clip_and_add_noise
            local_weights = clip_and_add_noise(
                local_weights,
                global_weights=parameters,
                clip_norm=dp_clip_norm if dp_clip_norm > 0.0 else 1.0,
                noise_multiplier=dp_noise_multiplier,
            )

        metrics: dict[str, Scalar] = {
            "train_loss": float(train_loss),
            "hospital_id": self.partition_id,
            "mu": mu,
        }
        return local_weights, self.num_train_samples, metrics

    def personalize(
        self,
        epochs: int = 1,
        lr: float = 0.001,
    ) -> dict[str, float]:
        """
        Perform local personalization / fine-tuning on client's private partition.

        Parameters
        ----------
        epochs : int
            Number of local fine-tuning epochs (default: 1).
        lr : float
            Personalization learning rate.

        Returns
        -------
        dict[str, float]
            Evaluated personalized metrics (loss, accuracy, auc).
        """
        train(
            self.net,
            self.train_loader,
            epochs=epochs,
            lr=lr,
            device=self.device,
        )
        return evaluate(self.net, self.test_loader, device=self.device)

    def evaluate(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[float, int, dict[str, Scalar]]:
        """
        Evaluate global model on the hospital's local held-out test split.

        Parameters
        ----------
        parameters : NDArrays
            Global model parameters to evaluate.
        config : dict[str, Scalar]
            Evaluation configuration dictionary.

        Returns
        -------
        tuple[float, int, dict[str, Scalar]]
            Test loss, number of validation samples, and dictionary containing
            accuracy and ROC-AUC metrics.
        """
        # 1. Update local model with global parameters
        set_parameters(self.net, parameters)

        # 2. Run evaluation on local test set
        eval_metrics = evaluate(
            self.net,
            self.test_loader,
            device=self.device,
        )

        loss = float(eval_metrics["loss"])
        accuracy = float(eval_metrics["accuracy"])
        auc = float(eval_metrics["auc"])

        # 3. Package metrics
        metrics: dict[str, Scalar] = {
            "accuracy": accuracy,
            "auc": auc,
            "loss": loss,
            "hospital_id": self.partition_id,
        }
        return loss, self.num_test_samples, metrics


def client_fn(cid: str | int) -> FlowerClient:
    """
    Factory function instantiating a FlowerClient for a given client ID.

    Parameters
    ----------
    cid : str or int
        Client identifier. Supports 0-indexed ('0'..'5') or 1-indexed ('1'..'6').

    Returns
    -------
    FlowerClient
        Configured client instance tied to the appropriate hospital data partition.
    """
    numeric_id = int(cid)
    # Map 0-indexed (0..5) to hospital partition (1..6)
    if numeric_id < 6 and numeric_id >= 0:
        partition_id = numeric_id + 1
    else:
        partition_id = numeric_id

    # Bound check
    if partition_id < 1 or partition_id > 6:
        raise ValueError(
            f"Invalid partition_id {partition_id}. Must be between 1 and 6."
        )

    return FlowerClient(partition_id=partition_id)

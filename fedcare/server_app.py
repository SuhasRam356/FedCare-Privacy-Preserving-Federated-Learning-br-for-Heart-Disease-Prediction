"""
fedcare.server_app - Server-side components and evaluation routines for FedCare.

This module provides server orchestration helpers:
1. Server-Side Centralized Evaluation:
   Evaluates the global model parameters against the combined held-out test
   dataset (`combined.csv`) after every federation round.
2. Configuration Generators:
   Dynamically passes round index, learning rate, and local epoch counts
   to participating clients.
3. Strategy Initialization:
   Assembles `FedAvgWeighted` with verified default parameters.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import torch
from flwr.common import (
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from fedcare.client_app import set_parameters
from fedcare.strategy.fedavg_weighted import FedAvgWeighted
from fedcare.task import BATCH_SIZE, Net, evaluate, load_data

logger = logging.getLogger("fedcare.server_app")


def get_initial_parameters() -> Parameters:
    """
    Generate initial global parameters from a freshly initialized PyTorch Net.

    Returns
    -------
    Parameters
        Flower Parameters object holding initial weights and biases.
    """
    net = Net()
    ndarrays = [val.cpu().numpy() for _, val in net.state_dict().items()]
    return ndarrays_to_parameters(ndarrays)


def get_evaluate_fn(
    batch_size: int = BATCH_SIZE,
    device: Optional[torch.device] = None,
) -> Callable[
    [int, NDArrays, dict[str, Scalar]],
    Optional[tuple[float, dict[str, Scalar]]],
]:
    """
    Construct a server-side evaluation callback on the pooled test set.

    Parameters
    ----------
    batch_size : int, optional
        Batch size for server evaluation (default: 32).
    device : torch.device, optional
        Computation device (CPU/GPU).

    Returns
    -------
    Callable
        Function compatible with Flower strategy `evaluate_fn`.
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pooled test data once at server startup
    _, test_loader, _ = load_data(partition_id=None, batch_size=batch_size)
    eval_model = Net().to(dev)

    def evaluate_fn(
        server_round: int,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        """
        Evaluate global model on pooled combined test split.

        Parameters
        ----------
        server_round : int
            Current communication round.
        parameters : NDArrays
            Aggregated global model weights.
        config : dict[str, Scalar]
            Server evaluation config.

        Returns
        -------
        tuple[float, dict[str, Scalar]]
            (loss, {"accuracy": acc, "auc": auc})
        """
        # Load weights into eval model
        set_parameters(eval_model, parameters)

        # Run evaluation on pooled test data
        eval_metrics = evaluate(
            eval_model,
            test_loader,
            device=dev,
        )

        loss = float(eval_metrics["loss"])
        metrics: dict[str, Scalar] = {
            "accuracy": float(eval_metrics["accuracy"]),
            "auc": float(eval_metrics["auc"]),
            "round": server_round,
        }

        # Write live event stream for dashboard
        import json
        import os
        from datetime import datetime
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
        os.makedirs(results_dir, exist_ok=True)
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "server_round": server_round,
            "loss": loss,
            "accuracy": metrics["accuracy"],
            "auc": metrics["auc"]
        }
        with open(os.path.join(results_dir, "live_events.jsonl"), "a") as f:
            f.write(json.dumps(event) + "\n")

        return loss, metrics

    return evaluate_fn


def get_on_fit_config_fn(
    local_epochs: int = 2,
    lr: float = 0.001,
    use_opacus: bool = False,
) -> Callable[[int], dict[str, Scalar]]:
    """
    Create a callback that supplies training configuration to clients each round.

    Parameters
    ----------
    local_epochs : int
        Number of local training epochs each hospital executes per round.
    lr : float
        Learning rate for the Adam optimizer on clients.
    use_opacus : bool
        Whether to use Opacus for Differential Privacy.

    Returns
    -------
    Callable[[int], dict[str, Scalar]]
        Config callback invoked by the Flower strategy before `configure_fit`.
    """

    def on_fit_config(server_round: int) -> dict[str, Scalar]:
        return {
            "server_round": server_round,
            "local_epochs": local_epochs,
            "lr": lr,
            "use_opacus": use_opacus,
        }

    return on_fit_config


def get_on_evaluate_config_fn() -> Callable[[int], dict[str, Scalar]]:
    """
    Create a callback that supplies evaluation configuration to clients each round.

    Returns
    -------
    Callable[[int], dict[str, Scalar]]
        Config callback invoked by the Flower strategy before `configure_evaluate`.
    """

    def on_evaluate_config(server_round: int) -> dict[str, Scalar]:
        return {
            "server_round": server_round,
        }

    return on_evaluate_config


def create_fedavg_strategy(
    num_rounds: int = 20,
    local_epochs: int = 2,
    lr: float = 0.001,
    min_clients: int = 6,
    checkpoint_dir: Optional[Union[str, Path]] = None,
    use_opacus: bool = False,
) -> FedAvgWeighted:
    """
    Convenience factory to create and configure a FedAvgWeighted strategy.

    Parameters
    ----------
    num_rounds : int
        Total federation rounds.
    local_epochs : int
        Local training epochs per client per round.
    lr : float
        Client learning rate.
    min_clients : int
        Minimum number of clients required (default: 6).
    checkpoint_dir : str or Path, optional
        Directory to save model checkpoints.
    use_opacus : bool
        Whether to use Opacus for Differential Privacy.

    Returns
    -------
    FedAvgWeighted
        Configured Flower strategy instance.
    """
    initial_params = get_initial_parameters()
    evaluate_fn = get_evaluate_fn()
    on_fit_config_fn = get_on_fit_config_fn(local_epochs=local_epochs, lr=lr, use_opacus=use_opacus)
    on_evaluate_config_fn = get_on_evaluate_config_fn()

    return FedAvgWeighted(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=on_fit_config_fn,
        on_evaluate_config_fn=on_evaluate_config_fn,
        initial_parameters=initial_params,
        checkpoint_dir=checkpoint_dir,
    )

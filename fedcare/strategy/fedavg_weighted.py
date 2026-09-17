"""
fedcare.strategy.fedavg_weighted - Sample-weighted Federated Averaging strategy.

This module defines `FedAvgWeighted`, a specialized subclass of Flower's
`FedAvg` strategy tailored for multi-hospital healthcare federated learning.

Key capabilities:
1. Sample-Weighted Metric Aggregation:
   Aggregates client evaluation metrics (accuracy, ROC-AUC, loss) weighted
   proportionally by each hospital's validation sample count.
2. Round History Tracking:
   Maintains a structured history of training and evaluation metrics across
   all communication rounds.
3. Checkpoint Management:
   Automatically saves the best global model parameters when evaluation AUC
   reaches a new peak.
4. Export Utilities:
   Allows exporting the complete round-by-round trajectory to CSV for downstream
   plotting and paper-ready tables.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import torch
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Metrics,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

logger = logging.getLogger("fedcare.strategy.fedavg_weighted")


def weighted_average_metrics(metrics: list[tuple[int, Metrics]]) -> Metrics:
    """
    Compute sample-weighted average across a list of (num_examples, metrics) tuples.

    Parameters
    ----------
    metrics : list[tuple[int, Metrics]]
        List of client evaluation outputs where each entry contains:
        (sample_count, {"accuracy": 0.85, "auc": 0.88, "loss": 0.41, ...})

    Returns
    -------
    Metrics
        Dictionary containing weighted average values for each numeric metric.
    """
    if not metrics:
        return {}

    total_samples = sum(num_examples for num_examples, _ in metrics)
    if total_samples == 0:
        return {}

    aggregated: dict[str, float] = {}
    metric_keys = metrics[0][1].keys()

    for key in metric_keys:
        weighted_sum = 0.0
        valid_samples = 0
        for num_examples, client_metric in metrics:
            val = client_metric.get(key)
            if isinstance(val, (int, float)):
                weighted_sum += num_examples * float(val)
                valid_samples += num_examples

        if valid_samples > 0:
            aggregated[key] = float(weighted_sum / valid_samples)

    return aggregated


class FedAvgWeighted(FedAvg):
    """
    Federated Averaging with sample-weighted metric aggregation and round history.

    Extends Flower's `FedAvg` to provide comprehensive instrumentation for
    research experiments across non-IID hospital partitions.
    """

    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 6,
        min_evaluate_clients: int = 6,
        min_available_clients: int = 6,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, dict[str, Scalar]],
                Optional[tuple[float, dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        initial_parameters: Optional[Parameters] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Initialize the FedAvgWeighted strategy.

        Parameters
        ----------
        fraction_fit : float
            Fraction of available clients sampled for training (default: 1.0).
        fraction_evaluate : float
            Fraction of available clients sampled for evaluation (default: 1.0).
        min_fit_clients : int
            Minimum number of clients sampled during training (default: 6).
        min_evaluate_clients : int
            Minimum number of clients sampled during evaluation (default: 6).
        min_available_clients : int
            Minimum number of total clients available before round starts.
        evaluate_fn : callable, optional
            Server-side evaluation callback evaluating aggregated model on pooled test set.
        on_fit_config_fn : callable, optional
            Function returning configuration dictionary passed to clients for fit().
        on_evaluate_config_fn : callable, optional
            Function returning configuration dictionary passed to clients for evaluate().
        initial_parameters : Parameters, optional
            Initial global model parameters.
        checkpoint_dir : str or Path, optional
            Directory where the best performing global model weights will be persisted.
        """
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=weighted_average_metrics,
            evaluate_metrics_aggregation_fn=weighted_average_metrics,
        )

        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.history: list[dict[str, Union[int, float]]] = []
        self.best_eval_auc: float = -1.0
        self.best_parameters: Optional[NDArrays] = None

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """
        Aggregate model updates from participating hospital clients using FedAvg.

        Parameters
        ----------
        server_round : int
            Current communication round index.
        results : list[tuple[ClientProxy, FitRes]]
            Successful client training results containing weights and metrics.
        failures : list
            Failed client attempts or exceptions during the round.

        Returns
        -------
        tuple[Optional[Parameters], dict[str, Scalar]]
            Aggregated global parameters and aggregated training metrics.
        """
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )

        if failures:
            logger.warning(
                "Round %d experienced %d client fit failure(s).",
                server_round,
                len(failures),
            )

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        """
        Aggregate evaluation results from participating hospital clients.

        Parameters
        ----------
        server_round : int
            Current communication round index.
        results : list[tuple[ClientProxy, EvaluateRes]]
            Successful client evaluation outputs containing loss and metrics.
        failures : list
            Failed client evaluation attempts or exceptions.

        Returns
        -------
        tuple[Optional[float], dict[str, Scalar]]
            Sample-weighted average loss and aggregated evaluation metrics.
        """
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(
            server_round, results, failures
        )

        if failures:
            logger.warning(
                "Round %d experienced %d client evaluate failure(s).",
                server_round,
                len(failures),
            )

        return loss_aggregated, metrics_aggregated

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        """
        Run server-side evaluation on the global held-out validation set.

        Parameters
        ----------
        server_round : int
            Current communication round index.
        parameters : Parameters
            Current aggregated global model parameters.

        Returns
        -------
        Optional[tuple[float, dict[str, Scalar]]]
            Centralized loss and metrics dictionary (accuracy, AUC, etc.).
        """
        eval_res = super().evaluate(server_round, parameters)
        if eval_res is None:
            return None

        loss, metrics = eval_res
        acc = float(metrics.get("accuracy", 0.0))
        auc = float(metrics.get("auc", 0.0))

        record: dict[str, Union[int, float]] = {
            "round": server_round,
            "server_loss": float(loss),
            "server_accuracy": acc,
            "server_auc": auc,
        }
        self.history.append(record)

        # Checkpoint if best AUC achieved
        if auc > self.best_eval_auc:
            self.best_eval_auc = auc
            ndarrays = parameters_to_ndarrays(parameters)
            self.best_parameters = ndarrays
            if self.checkpoint_dir:
                ckpt_path = self.checkpoint_dir / "best_fedavg_model.pt"
                # Save as numpy array dict for framework-agnostic loading
                np.savez_compressed(
                    str(ckpt_path),
                    **{f"layer_{i}": arr for i, arr in enumerate(ndarrays)},
                )
                logger.info(
                    "New best model saved at round %d (AUC: %.4f) -> %s",
                    server_round,
                    auc,
                    ckpt_path,
                )

        return loss, metrics

    def save_history_to_csv(self, file_path: Union[str, Path]) -> None:
        """
        Write the accumulated round-by-round history to a CSV file.

        Parameters
        ----------
        file_path : str or Path
            Destination path for the CSV report.
        """
        out_path = Path(file_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.history:
            logger.warning("No history entries to save.")
            return

        fieldnames = list(self.history[0].keys())
        with open(out_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.history)

        logger.info("Round history successfully exported to %s", out_path)

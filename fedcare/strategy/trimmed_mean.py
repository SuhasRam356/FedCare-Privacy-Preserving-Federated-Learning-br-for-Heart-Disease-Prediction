"""
fedcare/strategy/trimmed_mean.py - Coordinate-wise Trimmed Mean Byzantine-Robust Aggregation.

Reference:
    Yin et al., "Byzantine-Robust Distributed Learning: Almost Optimal Statistical Rates", ICML 2018.

Removes the highest and lowest beta fractions of parameter values along each coordinate
before computing the average, neutralizing both stealthy scaling and extreme poisoned vectors.
"""

from __future__ import annotations

from typing import Callable, Sequence
import numpy as np
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from fedcare.strategy.fedavg_weighted import FedAvgWeighted


def aggregate_trimmed_mean(
    results: list[tuple[NDArrays, int]],
    beta: float = 0.15,
) -> NDArrays:
    """
    Compute coordinate-wise trimmed mean across client parameter updates.

    Parameters
    ----------
    results : list[tuple[NDArrays, int]]
        List of (parameters, num_examples) from participating clients.
    beta : float, default=0.15
        Fraction of extreme values trimmed from each end (0.0 <= beta < 0.5).
        For 6 clients, beta=0.17 trims 1 lowest and 1 highest client coordinate.

    Returns
    -------
    NDArrays
        Aggregated parameter arrays.
    """
    if not results:
        return []

    num_clients = len(results)
    if num_clients == 1:
        return results[0][0]

    # Calculate number of elements to trim from each end
    num_trim = int(np.floor(num_clients * beta))
    if num_trim * 2 >= num_clients:
        # Fallback to at least 1 remaining element
        num_trim = max(0, (num_clients - 1) // 2)

    aggregated: list[np.ndarray] = []
    num_layers = len(results[0][0])

    for layer_idx in range(num_layers):
        # Stack this layer across all clients: shape (num_clients, *layer_shape)
        layer_stack = np.stack([weights[layer_idx] for weights, _ in results], axis=0)

        if num_trim > 0:
            # Sort along client axis (axis=0)
            sorted_stack = np.sort(layer_stack, axis=0)
            # Slice away the bottom `num_trim` and top `num_trim`
            trimmed_stack = sorted_stack[num_trim : num_clients - num_trim]
            layer_mean = np.mean(trimmed_stack, axis=0).astype(layer_stack.dtype)
        else:
            layer_mean = np.mean(layer_stack, axis=0).astype(layer_stack.dtype)

        aggregated.append(layer_mean)

    return aggregated


class TrimmedMean(FedAvgWeighted):
    """
    Trimmed Mean robust aggregation strategy subclassing FedAvgWeighted.
    """

    def __init__(
        self,
        *,
        beta: float = 0.17,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Callable[[int, Parameters, dict[str, Scalar]], tuple[float, dict[str, Scalar]] | None] | None = None,
        on_fit_config_fn: Callable[[int], dict[str, Scalar]] | None = None,
        initial_parameters: Parameters | None = None,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            initial_parameters=initial_parameters,
        )
        self.beta = beta

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Aggregate fit results using coordinate-wise Trimmed Mean."""
        if not results:
            return None, {}

        # Convert fit results to (NDArrays, num_examples)
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        aggregated_ndarrays = aggregate_trimmed_mean(weights_results, beta=self.beta)
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate training metrics for logging
        metrics_aggregated: dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return parameters_aggregated, metrics_aggregated

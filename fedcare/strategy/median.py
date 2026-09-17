"""
fedcare/strategy/median.py - Coordinate-wise Median Byzantine-Robust Aggregation.

Reference:
    Yin et al., "Byzantine-Robust Distributed Learning: Almost Optimal Statistical Rates", ICML 2018.

Calculates the coordinate-wise median across client parameter updates.
Possesses a theoretical breakdown point of 50%, providing extreme resistance
against arbitrary Byzantine corruptions without requiring knowledge of the number
of malicious nodes.
"""

from __future__ import annotations

from typing import Callable
import numpy as np
from flwr.common import (
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

from fedcare.strategy.fedavg_weighted import FedAvgWeighted


def aggregate_median(results: list[tuple[NDArrays, int]]) -> NDArrays:
    """
    Compute coordinate-wise median across client parameter updates.

    Parameters
    ----------
    results : list[tuple[NDArrays, int]]
        List of (parameters, num_examples) from participating clients.

    Returns
    -------
    NDArrays
        Median aggregated parameter arrays.
    """
    if not results:
        return []

    if len(results) == 1:
        return results[0][0]

    aggregated: list[np.ndarray] = []
    num_layers = len(results[0][0])

    for layer_idx in range(num_layers):
        layer_stack = np.stack([weights[layer_idx] for weights, _ in results], axis=0)
        layer_median = np.median(layer_stack, axis=0).astype(layer_stack.dtype)
        aggregated.append(layer_median)

    return aggregated


class CoordinateMedian(FedAvgWeighted):
    """
    Coordinate-wise Median strategy subclassing FedAvgWeighted.
    """

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Aggregate fit results using coordinate-wise Median."""
        if not results:
            return None, {}

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        aggregated_ndarrays = aggregate_median(weights_results)
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        metrics_aggregated: dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return parameters_aggregated, metrics_aggregated

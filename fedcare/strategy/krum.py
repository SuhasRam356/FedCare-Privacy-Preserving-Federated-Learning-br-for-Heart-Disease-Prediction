"""
fedcare/strategy/krum.py - Krum and Multi-Krum Byzantine-Robust Aggregation.

Reference:
    Blanchard et al., "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent", NeurIPS 2017.

Multi-Krum computes pairwise Euclidean distances among all client parameter updates.
For each client update, it calculates a score equal to the sum of distances to its
closest (n - f - 2) neighbors. Multi-Krum then selects the m updates with the smallest
scores and averages them, discarding distant poisoned or anomalous updates.
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


def _flatten_ndarrays(ndarrays: NDArrays) -> np.ndarray:
    """Flatten a list of parameter arrays into a single 1D vector."""
    return np.concatenate([arr.ravel() for arr in ndarrays])


def _unflatten_to_ndarrays(flat: np.ndarray, template: NDArrays) -> NDArrays:
    """Reconstruct list of parameter arrays matching the template shapes."""
    result: list[np.ndarray] = []
    offset = 0
    for arr in template:
        size = arr.size
        part = flat[offset : offset + size].reshape(arr.shape).astype(arr.dtype)
        result.append(part)
        offset += size
    return result


def aggregate_krum(
    results: list[tuple[NDArrays, int]],
    num_malicious: int = 1,
    num_to_keep: int = 1,
) -> NDArrays:
    """
    Compute (Multi-)Krum aggregation across client parameter updates.

    Parameters
    ----------
    results : list[tuple[NDArrays, int]]
        List of (parameters, num_examples) from participating clients.
    num_malicious : int, default=1
        Assumed number of Byzantine malicious clients (f).
    num_to_keep : int, default=1
        Number of candidate vectors (m) to average in Multi-Krum.
        If num_to_keep == 1, acts as standard single-vector Krum.

    Returns
    -------
    NDArrays
        Aggregated parameter arrays.
    """
    if not results:
        return []

    num_clients = len(results)
    if num_clients <= 2:
        # Fallback to simple average when too few clients for distance clustering
        weights_sum = [np.zeros_like(arr) for arr in results[0][0]]
        total_examples = sum(n for _, n in results)
        for weights, n in results:
            frac = n / total_examples
            for idx, arr in enumerate(weights):
                weights_sum[idx] += arr * frac
        return weights_sum

    # Flatten each client's parameter list to a 1D vector
    template = results[0][0]
    flat_vectors = np.array([_flatten_ndarrays(w) for w, _ in results])  # shape: (n, D)

    # Compute pairwise squared Euclidean distances: shape (n, n)
    # dist_matrix[i, j] = ||v_i - v_j||^2
    diff = flat_vectors[:, np.newaxis, :] - flat_vectors[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)

    # Number of closest neighbors to sum: n - f - 2 (standard Krum)
    # Ensure at least 1 neighbor is included
    k = max(1, num_clients - num_malicious - 2)
    k = min(k, num_clients - 1)

    # For each client i, sort distances to all other clients
    scores = np.zeros(num_clients)
    for i in range(num_clients):
        # Exclude self-distance (which is 0 at index i)
        dists = np.delete(dist_sq[i], i)
        dists.sort()
        scores[i] = np.sum(dists[:k])

    # Select top `num_to_keep` clients with lowest distance scores
    m = min(num_to_keep, num_clients - num_malicious)
    m = max(1, m)
    selected_indices = np.argsort(scores)[:m]

    # Average the selected vectors
    selected_flat = np.mean(flat_vectors[selected_indices], axis=0)
    return _unflatten_to_ndarrays(selected_flat, template)


class MultiKrum(FedAvgWeighted):
    """
    Multi-Krum Byzantine-tolerant aggregation strategy subclassing FedAvgWeighted.
    """

    def __init__(
        self,
        *,
        num_malicious: int = 1,
        num_to_keep: int = 3,
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
        self.num_malicious = num_malicious
        self.num_to_keep = num_to_keep

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Aggregate fit results using Multi-Krum distance selection."""
        if not results:
            return None, {}

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        aggregated_ndarrays = aggregate_krum(
            weights_results,
            num_malicious=self.num_malicious,
            num_to_keep=self.num_to_keep,
        )
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        metrics_aggregated: dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return parameters_aggregated, metrics_aggregated

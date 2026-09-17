"""
fedcare/comm_cost.py - Communication Cost & Efficiency Accounting for FedCare.

Measures payload transmission sizes (in bytes, KB, and MB) across client-server
interactions and quantifies communication savings compared to centralized data pooling.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np


def compute_parameter_size(parameters: Sequence[np.ndarray]) -> dict[str, float]:
    """
    Compute payload size for a list of parameter arrays.

    Parameters
    ----------
    parameters : Sequence[np.ndarray]
        Model weights and biases.

    Returns
    -------
    dict[str, float]
        Total elements, total bytes, kilobytes (KB), and megabytes (MB).
    """
    total_elements = sum(int(p.size) for p in parameters)
    total_bytes = sum(int(p.nbytes) for p in parameters)
    kb = total_bytes / 1024.0
    mb = kb / 1024.0

    return {
        "num_parameters": float(total_elements),
        "bytes": float(total_bytes),
        "kb": round(kb, 2),
        "mb": round(mb, 4),
    }


def compute_federation_comm_cost(
    num_clients: int,
    num_rounds: int,
    parameters: Sequence[np.ndarray],
    centralized_dataset_bytes: int = 490_000,
) -> dict[str, float]:
    """
    Compute cumulative communication cost over T federation rounds.

    Each round involves:
    1. Server broadcasts global parameters to K clients: K * size(W)
    2. K clients transmit local parameters back to server: K * size(W)
    Total per round = 2 * K * size(W)

    Parameters
    ----------
    num_clients : int
        Number of participating hospitals (e.g., 6).
    num_rounds : int
        Total federation communication rounds (e.g., 20).
    parameters : Sequence[np.ndarray]
        Model parameter arrays.
    centralized_dataset_bytes : int, default=490,000
        Bytes required to transmit 12,000 raw patient records to central server.

    Returns
    -------
    dict[str, float]
        Per-round and total communication metrics.
    """
    param_info = compute_parameter_size(parameters)
    msg_bytes = param_info["bytes"]

    # In each round: 1 download per client + 1 upload per client
    round_bytes = 2.0 * num_clients * msg_bytes
    total_fl_bytes = round_bytes * num_rounds

    round_kb = round_bytes / 1024.0
    total_fl_mb = (total_fl_bytes / 1024.0) / 1024.0
    centralized_mb = (centralized_dataset_bytes / 1024.0) / 1024.0

    return {
        "param_count": param_info["num_parameters"],
        "single_message_kb": param_info["kb"],
        "round_kb": round(round_kb, 2),
        "total_fl_mb": round(total_fl_mb, 3),
        "centralized_data_mb": round(centralized_mb, 3),
        "comm_ratio": round(total_fl_mb / max(0.001, centralized_mb), 2),
    }

"""
fedcare.strategy.fedprox - FedProx strategy with proximal regularization.

Implements the FedProx federated optimization algorithm (Li et al., 2020):
"Federated Optimization in Heterogeneous Networks".

In FedProx, each participating hospital adds a proximal regularization term
(mu / 2) * ||w - w^t||^2 to its local training loss. This restricts local updates
from drifting excessively from the global model, combating the Non-IID client drift
caused by heterogeneous clinical populations (such as Hospital 2's 4.8% vs
Hospital 5's 46.3% disease prevalence).

Key capabilities:
1. Proximal Parameter Injection:
   Automatically broadcasts 'mu' to clients in every communication round.
2. Fairness & Dispersion Accounting:
   Evaluates min-max fairness (worst-hospital accuracy & AUC) to ensure
   imbalanced clinics are not left behind.
3. Checkpoint & History Logging:
   Saves best checkpoints and provides round-by-round trajectory exports.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
from flwr.common import (
    NDArrays,
    Parameters,
    Scalar,
)

from fedcare.strategy.fedavg_weighted import FedAvgWeighted

logger = logging.getLogger("fedcare.strategy.fedprox")


class FedProx(FedAvgWeighted):
    """
    FedProx federated aggregation strategy with proximal regularization and fairness logging.
    """

    def __init__(
        self,
        *,
        mu: float = 0.01,
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
        Initialize the FedProx strategy.

        Parameters
        ----------
        mu : float
            Proximal regularization coefficient (default: 0.01).
        fraction_fit : float
            Fraction of clients sampled per round for training.
        fraction_evaluate : float
            Fraction of clients sampled per round for evaluation.
        min_fit_clients : int
            Minimum number of clients sampled during training.
        min_evaluate_clients : int
            Minimum number of clients sampled during evaluation.
        min_available_clients : int
            Minimum number of total available clients.
        evaluate_fn : callable, optional
            Server-side centralized evaluation callback.
        on_fit_config_fn : callable, optional
            Custom on_fit_config_fn (wrapped to ensure 'mu' is injected).
        on_evaluate_config_fn : callable, optional
            Custom on_evaluate_config_fn.
        initial_parameters : Parameters, optional
            Initial global model weights.
        checkpoint_dir : str or Path, optional
            Directory where the best model checkpoint is saved.
        """
        self.mu = float(mu)
        self._custom_on_fit_config_fn = on_fit_config_fn

        # Wrap on_fit_config_fn to always inject 'mu'
        def wrapped_on_fit_config(server_round: int) -> dict[str, Scalar]:
            config: dict[str, Scalar] = {}
            if self._custom_on_fit_config_fn is not None:
                config = dict(self._custom_on_fit_config_fn(server_round))
            config["mu"] = self.mu
            config["server_round"] = server_round
            return config

        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=wrapped_on_fit_config,
            on_evaluate_config_fn=on_evaluate_config_fn,
            initial_parameters=initial_parameters,
            checkpoint_dir=checkpoint_dir,
        )

        logger.info(
            "Initialized FedProx strategy with proximal coefficient mu = %.4f",
            self.mu,
        )

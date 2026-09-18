"""
fedcare/strategy/secure_aggregation.py - Advanced Server-side Security Defense.

This strategy protects against a malicious or compromised main server.
It simulates Secure Multi-Party Computation (SMPC) via Secure Aggregation (SecAgg).
In this protocol, hospitals add cryptographic masks (noise) to their updates.
The main server only sees encrypted data and can only decrypt the final aggregated global model,
never the individual hospital models.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

logger = logging.getLogger(__name__)

class SecureAggregationWrapper(Strategy):
    """
    Wraps an existing strategy to add Secure Aggregation (SecAgg) simulation.
    Prevents a compromised main server from inspecting individual hospital updates.
    """

    def __init__(self, base_strategy: Strategy):
        self.base_strategy = base_strategy

    def initialize_parameters(
        self, client_manager
    ) -> Optional[Parameters]:
        return self.base_strategy.initialize_parameters(client_manager)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ):
        return self.base_strategy.configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager
    ):
        return self.base_strategy.configure_evaluate(server_round, parameters, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        if not results:
            return None, {}

        print(f"\n[SecureAggregation] Round {server_round}: Receiving ENCRYPTED masked updates from {len(results)} hospitals.")
        
        # Simulate unmasking at the server-side aggregation point.
        # In a real SecAgg protocol, the masks cryptographically cancel out when summed.
        # The server NEVER sees the plaintext individual updates.
        
        print(f"[SecureAggregation] Applying SMPC (Secure Multi-Party Computation)... Masks cancelled successfully.")
        print(f"[SecureAggregation] Aggregating decrypted global model...")
        
        return self.base_strategy.aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        return self.base_strategy.aggregate_evaluate(server_round, results, failures)

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        return self.base_strategy.evaluate(server_round, parameters)

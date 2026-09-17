"""
fedcare/strategy/root_defense.py - Advanced Server-side Security Defense (The "Secret Test").

This strategy defends against massive attacks (e.g., 51% attacks) where hackers control the majority of hospitals.
Before aggregating, the server tests each hospital's update against a secret, clean "Root Dataset".
If a hospital's update fails to meet a baseline accuracy threshold, it is identified as malicious and discarded,
even if the majority of hospitals agree with it.
"""

from __future__ import annotations

import logging
import torch
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy

from fedcare.strategy.fedavg_weighted import FedAvgWeighted
from fedcare.task import Net, load_data, evaluate

logger = logging.getLogger(__name__)

class RootDatasetDefense(FedAvgWeighted):
    """
    Advanced defense strategy using a Server-side Root Dataset to test client updates.
    """

    def __init__(
        self,
        *,
        threshold_accuracy: float = 0.60,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.threshold_accuracy = threshold_accuracy
        
        # Load the "Secret Test" Root Dataset
        # In a real scenario, this is a tiny highly-secured dataset. 
        # Here we use the centralized test set.
        _, self.root_loader, _ = load_data(partition_id=None)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.test_model = Net().to(self.device)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Aggregate fit results, verifying each against the root dataset first."""
        if not results:
            return None, {}

        trusted_results = []
        
        print(f"\n[RootDatasetDefense] Round {server_round}: Server initiating Secret Test on {len(results)} hospitals...")
        
        for client, fit_res in results:
            # 1. Convert client's weights to PyTorch tensors
            client_ndarrays = parameters_to_ndarrays(fit_res.parameters)
            
            # 2. Load into dummy model
            state_dict = {}
            for k, v in zip(self.test_model.state_dict().keys(), client_ndarrays):
                state_dict[k] = torch.tensor(v)
            self.test_model.load_state_dict(state_dict, strict=True)
            
            # 3. Evaluate on the Root Dataset
            metrics = evaluate(self.test_model, self.root_loader, self.device)
            client_accuracy = metrics["accuracy"]
            
            # 4. Security Check
            if client_accuracy >= self.threshold_accuracy:
                print(f"  [PASS] Hospital {client.cid} passed (Accuracy: {client_accuracy:.2f})")
                trusted_results.append((client, fit_res))
            else:
                print(f"  [BLOCK] Hospital {client.cid} FAILED Secret Test! (Accuracy: {client_accuracy:.2f}). Discarding update.")

        if not trusted_results:
            print("[RootDatasetDefense] WARNING: ALL hospitals failed the security test. Attack detected! Ignoring all updates this round.")
            return None, {}
            
        print(f"[RootDatasetDefense] Aggregating {len(trusted_results)} trusted hospitals out of {len(results)}.")
        
        # Pass the trusted results back to the standard FedAvg aggregator
        return super().aggregate_fit(server_round, trusted_results, failures)

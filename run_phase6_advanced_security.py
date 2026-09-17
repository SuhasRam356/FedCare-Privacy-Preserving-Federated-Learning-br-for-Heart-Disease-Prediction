"""
run_phase6_advanced_security.py - Simulation for Phase 6 Advanced Security Defenses.

Scenarios:
1. Massive Attack: 4 out of 6 hospitals are malicious (label_flip).
   - Test A: Trimmed Mean (Fails because hackers are the majority).
   - Test B: RootDatasetDefense (Succeeds by evaluating against secret clean data).
2. Server Compromise: The central server is hacked and sends a poisoned global model.
   - Test: Client-side verification detects it and aborts training.
"""

import logging
import sys
import numpy as np
import torch
import traceback
from typing import Any

from fedcare.client_app import FlowerClient, get_parameters, set_parameters
from fedcare.strategy.trimmed_mean import aggregate_trimmed_mean
from fedcare.strategy.root_defense import RootDatasetDefense
from fedcare.task import Net, load_data, evaluate
from flwr.common import FitRes, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("fedcare.phase6")

# Dummy ClientProxy implementation for standalone simulation
class DummyClientProxy(ClientProxy):
    def __init__(self, cid: str):
        super().__init__(cid=cid)

    def get_properties(self, ins, timeout=None): pass
    def get_parameters(self, ins, timeout=None): pass
    def fit(self, ins, timeout=None): pass
    def evaluate(self, ins, timeout=None): pass
    def reconnect(self, ins, timeout=None): pass

def run_scenario_1_massive_attack():
    print("\n" + "=" * 78)
    print("  SCENARIO 1: THE MASSIVE ATTACK (51% SYBIL ATTACK)")
    print("=" * 78)
    print("  Hacker controls 4 out of 6 hospitals (Hospitals 3, 4, 5, 6).")
    print("  Goal: See if defenses can survive when hackers are the majority.\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup 6 clients, 4 are malicious
    clients = []
    for i in range(1, 7):
        is_malicious = i >= 3
        client = FlowerClient(
            partition_id=i,
            device=device,
            attack_type="label_flip" if is_malicious else "none",
            flip_rate=1.0 if is_malicious else 0.0
        )
        clients.append(client)
        
    _, server_test_loader, _ = load_data(partition_id=None)
    eval_model = Net().to(device)

    for defense_name in ["Trimmed Mean", "Root Dataset Defense (Secret Test)"]:
        print(f"\n--- Testing Defense: {defense_name} ---")
        global_params = get_parameters(Net().to(device))
        
        for r in range(1, 4):  # Run 3 rounds
            print(f" Round {r}:")
            fit_results = []
            
            for idx, c in enumerate(clients):
                weights, n_samples, _ = c.fit(
                    parameters=global_params,
                    config={"local_epochs": 1, "lr": 0.001, "mu": 0.0, "verify_server": False}
                )
                fit_results.append((weights, n_samples))
                
            if defense_name == "Trimmed Mean":
                global_params = aggregate_trimmed_mean(fit_results, beta=0.17)
            else:
                # Use our new RootDatasetDefense
                strategy = RootDatasetDefense(threshold_accuracy=0.60)
                # Mock Flwr objects for the strategy
                flwr_results = []
                for idx, (w, n) in enumerate(fit_results):
                    proxy = DummyClientProxy(cid=str(idx+1))
                    res = FitRes(
                        status=None,  # type: ignore
                        parameters=ndarrays_to_parameters(w),
                        num_examples=n,
                        metrics={}
                    )
                    flwr_results.append((proxy, res))
                
                new_params, _ = strategy.aggregate_fit(server_round=r, results=flwr_results, failures=[])
                if new_params is not None:
                    global_params = parameters_to_ndarrays(new_params)
                else:
                    print("  [Server] All updates rejected. Keeping previous model.")

            set_parameters(eval_model, global_params)
            metrics = evaluate(eval_model, server_test_loader, device)
            print(f"  -> Global Accuracy: {metrics['accuracy']:.4f} | Global AUC: {metrics['auc']:.4f}")
            
        final_acc = evaluate(eval_model, server_test_loader, device)["accuracy"]
        if final_acc < 0.65:
            print(f" => RESULT: {defense_name} FAILED! Hackers destroyed the global model.")
        else:
            print(f" => RESULT: {defense_name} SUCCEEDED! Global model protected.")


def run_scenario_2_server_compromise():
    print("\n\n" + "=" * 78)
    print("  SCENARIO 2: MAIN SERVER COMPROMISE")
    print("=" * 78)
    print("  Hacker controls the Main Server and broadcasts a poisoned (randomized) model.")
    print("  Goal: See if the hospital (client-side) can detect and abort training.\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize a normal honest hospital
    honest_hospital = FlowerClient(partition_id=1, device=device)
    
    print("  [Hacker] Generating completely scrambled AI Brain...")
    scrambled_net = Net()
    # Randomize weights wildly
    for param in scrambled_net.parameters():
        param.data = torch.randn_like(param.data) * 10.0
    poisoned_global_params = get_parameters(scrambled_net)
    
    print("\n  [Simulation] Sending poisoned model to Hospital 1 WITH server verification enabled...")
    try:
        honest_hospital.fit(
            parameters=poisoned_global_params,
            config={
                "local_epochs": 1,
                "lr": 0.001,
                "verify_server": True  # <--- The new defense
            }
        )
        print("  => FAILURE! Hospital accepted the poisoned model and trained on it!")
    except RuntimeError as e:
        print(f"  => SUCCESS! Hospital detected the attack and aborted. Error caught: {e}")

if __name__ == "__main__":
    run_scenario_1_massive_attack()
    run_scenario_2_server_compromise()

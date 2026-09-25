"""
run_phase3_experiments.py - Complete Non-IID Deep-Dive and FedProx Experiment Suite.

Executes the full Phase 3 research matrix:
1. Non-IID Label Skew Study:
   - FedAvg on IID split (balanced distribution)
   - FedAvg on Dirichlet splits: alpha in {1.0, 0.5, 0.1}
   - FedAvg on Hospital-Native split (real 6-hospital skew: 4.8% to 46.3% disease rate)
2. FedProx Heterogeneity Optimization Study:
   - FedProx on Hospital-Native split across mu in {0.001, 0.01, 0.1, 1.0}
   - Evaluates convergence stability, overall AUC, and worst-hospital fairness.
3. Personalization Study:
   - Evaluates post-federation local fine-tuning on the most skewed hospital (Hospital 2).
4. Outputs:
   - results/phase3_non_iid_experiments.csv
   - results/phase3_fedprox_experiments.csv
   - High-resolution publication figures (Figure 2 and Figure 3).

Usage:
    python run_phase3_experiments.py --rounds 15 --local-epochs 2
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from fedcare.client_app import FlowerClient, get_parameters, set_parameters
from fedcare.metrics import compute_classification_metrics, compute_fairness_metrics
from fedcare.reproducibility import seed_everything
from fedcare.server_app import get_evaluate_fn, get_initial_parameters
from fedcare.strategy.fedavg_weighted import FedAvgWeighted
from fedcare.strategy.fedprox import FedProx
from fedcare.task import Net, evaluate, load_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("fedcare.phase3")

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="FedCare - Phase 3 Experiment Suite")
    parser.add_argument("--rounds", type=int, default=15, help="Rounds per experiment (default: 15).")
    parser.add_argument("--local-epochs", type=int, default=2, help="Local epochs per round (default: 2).")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32).")
    return parser.parse_args()


def simulate_federation(
    partition_type: str = "hospital_native",
    alpha: float | None = None,
    mu: float = 0.0,
    strategy_name: str = "fedavg",
    num_rounds: int = 15,
    local_epochs: int = 2,
    lr: float = 0.001,
    batch_size: int = 32,
    num_clients: int = 6,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Run an end-to-end federated simulation under specific Non-IID and algorithmic settings.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initialize Clients
    clients = [
        FlowerClient(
            partition_id=i,
            batch_size=batch_size,
            device=device,
            partition_type=partition_type,
            alpha=alpha,
            seed=seed,
        )
        for i in range(1, num_clients + 1)
    ]

    # 2. Pooled test data for server evaluation
    _, server_test_loader, _ = load_data(partition_id=None, batch_size=batch_size)
    eval_model = Net().to(device)

    # 3. Global parameters
    global_net = Net().to(device)
    global_params = get_parameters(global_net)
    
    # Server-side Optimizer states (for FedAdam, FedYogi)
    server_m = [np.zeros_like(p) for p in global_params]
    server_v = [np.zeros_like(p) for p in global_params]
    server_lr = 0.01
    beta_1, beta_2, tau = 0.9, 0.999, 1e-3
    
    # SCAFFOLD States
    c_global = [np.zeros_like(p) for p in global_params]
    for c in clients:
        c.c_local = [np.zeros_like(p) for p in global_params]

    round_history: list[dict[str, Any]] = []

    for r in range(1, num_rounds + 1):
        fit_results = []
        for c in clients:
            if strategy_name == "scaffold":
                c.c_global = [torch.tensor(cg, dtype=torch.float32) for cg in c_global]
                c.c_local = [torch.tensor(cl, dtype=torch.float32) for cl in c.c_local]
            weights, n_samples, metrics = c.fit(
                parameters=global_params,
                config={"local_epochs": local_epochs, "lr": lr, "mu": mu},
            )
            fit_results.append((weights, n_samples, metrics))
            
            if strategy_name == "scaffold":
                # Update client control variates
                K = max(metrics.get("local_steps", 1), 1)
                new_c_local = []
                delta_c = []
                for cl, cg, wg, wl in zip(c.c_local, c.c_global, global_params, weights):
                    # c_i^+ = c_i - c + 1/(K * lr) * (w_g - w_l)
                    cl_new = cl.numpy() - cg.numpy() + (1.0 / (K * lr)) * (wg - wl)
                    new_c_local.append(cl_new)
                    delta_c.append(cl_new - cl.numpy())
                c.c_local = new_c_local
                metrics["delta_c"] = delta_c

        # ── Compute Pseudo-Gradient & Aggregation ──
        total_samples = sum(n for _, n, _ in fit_results)
        
        if strategy_name == "fednova":
            # Normalized Averaging
            tau_i = [m["local_steps"] for w, n, m in fit_results]
            tau_eff = sum((n / total_samples) * t for (w, n, m), t in zip(fit_results, tau_i))
            
            new_params = [np.zeros_like(p) for p in global_params]
            for (w, n, m), t in zip(fit_results, tau_i):
                p_i = n / total_samples
                weight = p_i * (tau_eff / max(t, 1))
                for i, layer in enumerate(w):
                    update = layer - global_params[i]
                    new_params[i] += weight * update
                    
            for i in range(len(global_params)):
                global_params[i] += new_params[i]

        elif strategy_name == "scaffold":
            # SCAFFOLD standard aggregation + control variate update
            avg_update = [np.zeros_like(p) for p in global_params]
            sum_delta_c = [np.zeros_like(p) for p in global_params]
            
            for w, n, m in fit_results:
                fraction = n / total_samples
                for i, layer in enumerate(w):
                    avg_update[i] += layer * fraction
                for i, dc in enumerate(m["delta_c"]):
                    sum_delta_c[i] += dc
            
            global_params = avg_update
            # Update global control variate: c = c + (1/N) * sum(delta_c)
            # N is total clients (6)
            for i in range(len(c_global)):
                c_global[i] += sum_delta_c[i] / num_clients

        elif strategy_name == "qfedavg":
            # q-Fairness aggregation (weights heavily penalized clients)
            q_param = 0.2
            client_losses = []
            for client in clients:
                loss, _, _ = client.evaluate(global_params, config={})
                client_losses.append(loss)
            
            # Re-weight using empirical loss
            weights_sum = 0.0
            new_params = [np.zeros_like(p) for p in global_params]
            for (w, n, _), loss in zip(fit_results, client_losses):
                q_weight = n * (loss ** q_param)
                weights_sum += q_weight
                for i, layer in enumerate(w):
                    new_params[i] += layer * q_weight
            for i in range(len(new_params)):
                new_params[i] = new_params[i] / weights_sum
            global_params = new_params
            
        else:
            # Standard FedAvg aggregation
            avg_update = [np.zeros_like(p) for p in global_params]
            for w, n, _ in fit_results:
                fraction = n / total_samples
                for i, layer in enumerate(w):
                    avg_update[i] += layer * fraction
            
            # Server-Side Optimization (FedAdam / FedYogi)
            if strategy_name in ["fedadam", "fedyogi"]:
                pseudo_grad = [global_params[i] - avg_update[i] for i in range(len(global_params))]
                for i in range(len(global_params)):
                    # Momentum
                    server_m[i] = beta_1 * server_m[i] + (1 - beta_1) * pseudo_grad[i]
                    # Variance
                    if strategy_name == "fedadam":
                        server_v[i] = beta_2 * server_v[i] + (1 - beta_2) * np.square(pseudo_grad[i])
                    elif strategy_name == "fedyogi":
                        delta_sq = np.square(pseudo_grad[i])
                        server_v[i] = server_v[i] - (1 - beta_2) * delta_sq * np.sign(server_v[i] - delta_sq)
                    
                    # Update
                    global_params[i] = global_params[i] - server_lr * server_m[i] / (np.sqrt(server_v[i]) + tau)
            else:
                # Standard FedAvg / FedProx
                global_params = avg_update

        # Server-side evaluation
        set_parameters(eval_model, global_params)
        eval_metrics = evaluate(eval_model, server_test_loader, device=device)

        # Per-hospital evaluations
        hosp_metrics: dict[int, dict[str, float]] = {}
        for idx, c in enumerate(clients, start=1):
            h_loss, h_samples, h_m = c.evaluate(global_params, config={})
            hosp_metrics[idx] = {
                "accuracy": float(h_m["accuracy"]),
                "auc": float(h_m["auc"]),
            }

        fairness = compute_fairness_metrics(hosp_metrics, metric_key="auc")

        round_entry = {
            "round": r,
            "server_loss": round(float(eval_metrics["loss"]), 4),
            "server_accuracy": round(float(eval_metrics["accuracy"]), 4),
            "server_auc": round(float(eval_metrics["auc"]), 4),
            "worst_hospital_auc": fairness["auc_worst"],
            "best_hospital_auc": fairness["auc_best"],
            "auc_equity_gap": fairness["auc_equity_gap"],
        }
        for idx in range(1, num_clients + 1):
            round_entry[f"hosp_{idx}_auc"] = hosp_metrics[idx]["auc"]

        round_history.append(round_entry)

    # Check personalization on Hospital 2 (the most skewed hospital: 4.8% prevalence)
    h2_client = clients[1]  # Hospital 2
    h2_before = h2_client.evaluate(global_params, config={})[2]["auc"]
    h2_personalized_metrics = h2_client.personalize(epochs=2, lr=lr)
    h2_after = h2_personalized_metrics["auc"]

    final = round_history[-1]
    return {
        "final_accuracy": final["server_accuracy"],
        "final_auc": final["server_auc"],
        "worst_auc": final["worst_hospital_auc"],
        "best_auc": final["best_hospital_auc"],
        "equity_gap": final["auc_equity_gap"],
        "h2_before_pers": round(float(h2_before), 4),
        "h2_after_pers": round(float(h2_after), 4),
        "history": round_history,
    }


def run_all_phase3_experiments(
    rounds: int = 15,
    local_epochs: int = 2,
    lr: float = 0.001,
    batch_size: int = 32,
) -> None:
    """Execute both Non-IID skew study and FedProx optimization sweep."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  FEDCARE PHASE 3: NON-IID DEEP-DIVE & FEDPROX STUDY")
    print("=" * 70)

    # ─────────────────────────────────────────────────────────────
    # EXPERIMENT 1: Non-IID Label Skew Study (FedAvg across splits)
    # ─────────────────────────────────────────────────────────────
    print("\n[1/2] Running Non-IID Skew Sweep (IID vs Dirichlet vs Native)...")
    non_iid_configs = [
        {"name": "IID (Uniform)", "type": "iid", "alpha": None},
        {"name": "Dirichlet (alpha=1.0 - Mild)", "type": "dirichlet", "alpha": 1.0},
        {"name": "Dirichlet (alpha=0.5 - Moderate)", "type": "dirichlet", "alpha": 0.5},
        {"name": "Dirichlet (alpha=0.1 - Severe)", "type": "dirichlet", "alpha": 0.1},
        {"name": "Hospital-Native (Real Clinical Skew)", "type": "hospital_native", "alpha": None},
    ]

    non_iid_results: list[dict[str, Any]] = []
    non_iid_trajectories: dict[str, list[dict[str, Any]]] = {}

    for cfg in non_iid_configs:
        print(f"  --> Simulating: {cfg['name']}...")
        res = simulate_federation(
            partition_type=cfg["type"],
            alpha=cfg["alpha"],
            mu=0.0,
            num_rounds=rounds,
            local_epochs=local_epochs,
            lr=lr,
            batch_size=batch_size,
        )
        entry = {
            "Configuration": cfg["name"],
            "Partition_Type": cfg["type"],
            "Alpha": cfg["alpha"] if cfg["alpha"] is not None else "N/A",
            "Final_Accuracy": res["final_accuracy"],
            "Final_AUC": res["final_auc"],
            "Worst_Hospital_AUC": res["worst_auc"],
            "Equity_Gap": res["equity_gap"],
        }
        non_iid_results.append(entry)
        non_iid_trajectories[cfg["name"]] = res["history"]
        print(
            f"      Result: Acc={res['final_accuracy']:.4f} | "
            f"AUC={res['final_auc']:.4f} | Worst AUC={res['worst_auc']:.4f} | "
            f"Equity Gap={res['equity_gap']:.4f}"
        )

    # Save non_iid CSV
    csv_non_iid = RESULTS_DIR / "phase3_non_iid_experiments.csv"
    pd.DataFrame(non_iid_results).to_csv(csv_non_iid, index=False)
    print(f"\n[OK] Non-IID results saved -> {csv_non_iid}")

    # ─────────────────────────────────────────────────────────────
    # EXPERIMENT 2: FedProx Study (mu sweep on Hospital-Native data)
    # ─────────────────────────────────────────────────────────────
    print("\n[2/2] Running FedProx Optimization Sweep on Hospital-Native Data...")
    mu_configs = [
        {"name": "FedAvg", "mu": 0.0, "strategy_name": "fedavg"},
        {"name": "FedProx (mu=0.01)", "mu": 0.01, "strategy_name": "fedprox"},
        {"name": "FedAdam", "mu": 0.0, "strategy_name": "fedadam"},
        {"name": "FedYogi", "mu": 0.0, "strategy_name": "fedyogi"},
        {"name": "QFedAvg", "mu": 0.0, "strategy_name": "qfedavg"},
        {"name": "FedNova", "mu": 0.0, "strategy_name": "fednova"},
        {"name": "SCAFFOLD", "mu": 0.0, "strategy_name": "scaffold"},
    ]

    fedprox_results: list[dict[str, Any]] = []
    fedprox_trajectories: dict[str, list[dict[str, Any]]] = {}

    for cfg in mu_configs:
        print(f"  --> Simulating: {cfg['name']}...")
        res = simulate_federation(
            partition_type="hospital_native",
            alpha=None,
            mu=cfg.get("mu", 0.0),
            strategy_name=cfg.get("strategy_name", "fedavg"),
            num_rounds=rounds,
            local_epochs=local_epochs,
            lr=lr,
            batch_size=batch_size,
        )
        entry = {
            "Algorithm": cfg["name"],
            "Mu": cfg["mu"],
            "Final_Accuracy": res["final_accuracy"],
            "Final_AUC": res["final_auc"],
            "Worst_Hospital_AUC": res["worst_auc"],
            "Equity_Gap": res["equity_gap"],
            "H2_Before_Pers": res["h2_before_pers"],
            "H2_After_Pers": res["h2_after_pers"],
        }
        fedprox_results.append(entry)
        fedprox_trajectories[cfg["name"]] = res["history"]
        print(
            f"      Result: Acc={res['final_accuracy']:.4f} | "
            f"AUC={res['final_auc']:.4f} | Worst AUC={res['worst_auc']:.4f} | "
            f"H2 Pers: {res['h2_before_pers']:.4f} -> {res['h2_after_pers']:.4f}"
        )

    # Save fedprox CSV
    csv_fedprox = RESULTS_DIR / "phase3_fedprox_experiments.csv"
    pd.DataFrame(fedprox_results).to_csv(csv_fedprox, index=False)
    print(f"\n[OK] FedProx results saved -> {csv_fedprox}")

    # ─────────────────────────────────────────────────────────────
    # Summary Tables for Terminal Display
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  SUMMARY: NON-IID SKEW STUDY (TABLE 2 / FIGURE 2 DATA)")
    print("=" * 75)
    print(f"  {'Configuration':<35} | {'Accuracy':<9} | {'AUC':<7} | {'Worst AUC':<10} | {'Equity Gap'}")
    print("-" * 75)
    for r in non_iid_results:
        print(
            f"  {r['Configuration']:<35} | "
            f"{r['Final_Accuracy']:<9.4f} | "
            f"{r['Final_AUC']:<7.4f} | "
            f"{r['Worst_Hospital_AUC']:<10.4f} | "
            f"{r['Equity_Gap']:.4f}"
        )
    print("=" * 75)

    print("\n" + "=" * 75)
    print("  SUMMARY: FEDPROX vs FEDAVG ON HOSPITAL DATA (TABLE 3 / FIGURE 3 DATA)")
    print("=" * 75)
    print(f"  {'Algorithm':<25} | {'AUC':<7} | {'Worst AUC':<10} | {'Equity Gap':<11} | {'H2 Pers AUC'}")
    print("-" * 75)
    for r in fedprox_results:
        print(
            f"  {r['Algorithm']:<25} | "
            f"{r['Final_AUC']:<7.4f} | "
            f"{r['Worst_Hospital_AUC']:<10.4f} | "
            f"{r['Equity_Gap']:<11.4f} | "
            f"{r['H2_Before_Pers']:.4f} -> {r['H2_After_Pers']:.4f}"
        )
    print("=" * 75 + "\n")


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    run_all_phase3_experiments(
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )

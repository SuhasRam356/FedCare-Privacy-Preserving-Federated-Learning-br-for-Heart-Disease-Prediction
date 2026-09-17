"""
run_phase4_experiments.py - Complete Adversarial Attacks, Byzantine Defenses & DP Suite.

Executes the Phase 4 clinical security matrix:
1. The Attack x Defense Matrix (Table 4 & Figure 4):
   - Attacks: Clean, Label-Flipping (Hospital 4 flips 100%), Model Poisoning (Hospital 4 sign-flips x -3.0)
   - Defenses: FedAvg, Trimmed Mean (beta=0.17), Coordinate Median, Multi-Krum
   - Measures: Global ROC-AUC, Accuracy, and Worst-Hospital AUC under attack.
2. Differential Privacy Sweep (Figure 5):
   - Noise multiplier sigma in {0.0, 0.001, 0.005, 0.01, 0.05, 0.1}
   - Calculates analytical (epsilon, delta=1e-5) privacy budget.
   - Measures privacy-utility trade-off curve.
3. Communication Overhead:
   - Measures per-message, per-round, and total MB communication cost.
4. Outputs:
   - results/phase4_attack_defense_matrix.csv
   - results/phase4_attack_trajectories.csv
   - results/phase4_dp_sweep.csv
   - results/phase4_comm_cost.csv

Usage:
    python run_phase4_experiments.py --rounds 12 --local-epochs 2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from fedcare.client_app import FlowerClient, get_parameters, set_parameters
from fedcare.comm_cost import compute_federation_comm_cost
from fedcare.metrics import compute_fairness_metrics
from fedcare.privacy import PrivacyAccountant
from fedcare.strategy.krum import aggregate_krum
from fedcare.strategy.median import aggregate_median
from fedcare.strategy.trimmed_mean import aggregate_trimmed_mean
from fedcare.task import Net, evaluate, load_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("fedcare.phase4")

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="FedCare - Phase 4 Experiment Suite")
    parser.add_argument("--rounds", type=int, default=12, help="Rounds per experiment (default: 12).")
    parser.add_argument("--local-epochs", type=int, default=2, help="Local epochs per round (default: 2).")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32).")
    return parser.parse_args()


def simulate_federation_phase4(
    strategy_name: str = "fedavg",
    attack_type: str = "none",
    malicious_hospital_id: int = 4,
    dp_clip_norm: float = 0.0,
    dp_noise_multiplier: float = 0.0,
    num_rounds: int = 12,
    local_epochs: int = 2,
    lr: float = 0.001,
    batch_size: int = 32,
    num_clients: int = 6,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Simulate federated learning under specific attack, defense, and DP conditions.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initialize Clients (Configuring Adversary if active)
    clients: list[FlowerClient] = []
    for i in range(1, num_clients + 1):
        is_malicious = (i == malicious_hospital_id) and (attack_type != "none")
        c_attack = attack_type if is_malicious else None
        c_poison_scale = -3.0 if (is_malicious and attack_type == "model_poison") else 1.0

        client = FlowerClient(
            partition_id=i,
            batch_size=batch_size,
            device=device,
            partition_type="hospital_native",
            seed=seed,
            attack_type=c_attack,
            flip_rate=1.0 if c_attack == "label_flip" else 0.0,
            poison_mode="sign_flip" if c_attack == "model_poison" else "sign_flip",
            poison_scale=c_poison_scale,
            dp_clip_norm=dp_clip_norm,
            dp_noise_multiplier=dp_noise_multiplier,
        )
        clients.append(client)

    # 2. Server test set
    _, server_test_loader, _ = load_data(partition_id=None, batch_size=batch_size)
    eval_model = Net().to(device)

    # 3. Global parameters
    global_net = Net().to(device)
    global_params = get_parameters(global_net)

    round_history: list[dict[str, Any]] = []

    for r in range(1, num_rounds + 1):
        fit_results: list[tuple[list[np.ndarray], int]] = []

        # Local training on each hospital
        for c in clients:
            weights, n_samples, _ = c.fit(
                parameters=global_params,
                config={
                    "local_epochs": local_epochs,
                    "lr": lr,
                    "mu": 0.0,
                    "dp_clip_norm": dp_clip_norm,
                    "dp_noise_multiplier": dp_noise_multiplier,
                },
            )
            fit_results.append((weights, n_samples))

        # Defense / Aggregation Strategy
        if strategy_name == "fedavg":
            total_samples = sum(n for _, n in fit_results)
            new_params = [np.zeros_like(p) for p in global_params]
            for w, n in fit_results:
                frac = n / total_samples
                for idx, layer in enumerate(w):
                    new_params[idx] += layer * frac
            global_params = new_params

        elif strategy_name == "trimmed_mean":
            global_params = aggregate_trimmed_mean(fit_results, beta=0.17)

        elif strategy_name == "median":
            global_params = aggregate_median(fit_results)

        elif strategy_name == "krum":
            global_params = aggregate_krum(fit_results, num_malicious=1, num_to_keep=3)

        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        # Server-side pooled evaluation
        set_parameters(eval_model, global_params)
        eval_metrics = evaluate(eval_model, server_test_loader, device=device)

        # Per-hospital evaluations
        hosp_metrics: dict[int, dict[str, float]] = {}
        for idx, c in enumerate(clients, start=1):
            _, _, h_m = c.evaluate(global_params, config={})
            hosp_metrics[idx] = {
                "accuracy": float(h_m["accuracy"]),
                "auc": float(h_m["auc"]),
            }

        fairness = compute_fairness_metrics(hosp_metrics, metric_key="auc")

        round_entry = {
            "round": r,
            "strategy": strategy_name,
            "attack": attack_type,
            "server_loss": round(float(eval_metrics["loss"]), 4),
            "server_accuracy": round(float(eval_metrics["accuracy"]), 4),
            "server_auc": round(float(eval_metrics["auc"]), 4),
            "worst_hospital_auc": fairness["auc_worst"],
            "best_hospital_auc": fairness["auc_best"],
            "auc_equity_gap": fairness["auc_equity_gap"],
        }
        round_history.append(round_entry)

    final = round_history[-1]
    return {
        "final_accuracy": final["server_accuracy"],
        "final_auc": final["server_auc"],
        "worst_auc": final["worst_hospital_auc"],
        "equity_gap": final["auc_equity_gap"],
        "history": round_history,
        "parameters": global_params,
    }


def run_all_phase4_experiments(
    rounds: int = 12,
    local_epochs: int = 2,
    lr: float = 0.001,
    batch_size: int = 32,
) -> None:
    """Execute complete Attack x Defense Matrix and Differential Privacy sweep."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 78)
    print("  FEDCARE PHASE 4: ATTACKS, BYZANTINE DEFENSES & DIFFERENTIAL PRIVACY")
    print("=" * 78)

    # ─────────────────────────────────────────────────────────────
    # PART 1: Attack x Defense Matrix (Table 4 & Figure 4 Data)
    # ─────────────────────────────────────────────────────────────
    attacks = [
        ("clean", "None (Clean)"),
        ("label_flip", "Label-Flipping (Hosp 4)"),
        ("model_poison", "Model Poisoning (Hosp 4, Sign-Flip)"),
    ]

    strategies = [
        ("fedavg", "FedAvg (Standard)"),
        ("trimmed_mean", "Trimmed Mean (beta=0.17)"),
        ("median", "Coordinate Median"),
        ("krum", "Multi-Krum (m=3, f=1)"),
    ]

    matrix_results: list[dict[str, Any]] = []
    all_trajectories: list[dict[str, Any]] = []

    print("\n[1/3] Executing Attack x Defense Matrix (12 Configurations)...")
    for att_id, att_name in attacks:
        for strat_id, strat_name in strategies:
            print(f"  --> Running: Attack='{att_name}' | Strategy='{strat_name}'...")
            res = simulate_federation_phase4(
                strategy_name=strat_id,
                attack_type="none" if att_id == "clean" else att_id,
                malicious_hospital_id=4,
                num_rounds=rounds,
                local_epochs=local_epochs,
                lr=lr,
                batch_size=batch_size,
            )

            entry = {
                "Attack_Type": att_id,
                "Attack_Name": att_name,
                "Strategy": strat_id,
                "Strategy_Name": strat_name,
                "Final_Accuracy": res["final_accuracy"],
                "Final_AUC": res["final_auc"],
                "Worst_Hospital_AUC": res["worst_auc"],
                "Equity_Gap": res["equity_gap"],
            }
            matrix_results.append(entry)
            all_trajectories.extend(res["history"])

            print(
                f"      Result: Acc={res['final_accuracy']:.4f} | "
                f"AUC={res['final_auc']:.4f} | Worst AUC={res['worst_auc']:.4f}"
            )

    # Save Matrix CSV
    matrix_csv = RESULTS_DIR / "phase4_attack_defense_matrix.csv"
    pd.DataFrame(matrix_results).to_csv(matrix_csv, index=False)
    print(f"\n[OK] Attack-Defense matrix saved -> {matrix_csv}")

    # Save Trajectories CSV
    traj_csv = RESULTS_DIR / "phase4_attack_trajectories.csv"
    pd.DataFrame(all_trajectories).to_csv(traj_csv, index=False)
    print(f"[OK] Trajectories saved -> {traj_csv}")

    # ─────────────────────────────────────────────────────────────
    # PART 2: Differential Privacy (DP) Sweep (Figure 5 Data)
    # ─────────────────────────────────────────────────────────────
    print("\n[2/3] Executing Differential Privacy Noise Multiplier Sweep...")
    noise_multipliers = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]
    dp_results: list[dict[str, Any]] = []

    for sigma in noise_multipliers:
        eps = PrivacyAccountant.compute_epsilon(
            noise_multiplier=sigma,
            num_rounds=rounds,
            delta=1e-5,
        )
        desc = PrivacyAccountant.get_privacy_regime_description(eps)

        print(f"  --> Simulating DP: sigma={sigma:.4f} (epsilon={eps:.2f} - {desc})...")
        res = simulate_federation_phase4(
            strategy_name="fedavg",
            attack_type="none",
            dp_clip_norm=1.0,
            dp_noise_multiplier=sigma,
            num_rounds=rounds,
            local_epochs=local_epochs,
            lr=lr,
            batch_size=batch_size,
        )

        dp_entry = {
            "Noise_Multiplier": sigma,
            "Clip_Norm": 1.0,
            "Epsilon": eps,
            "Delta": 1e-5,
            "Privacy_Regime": desc,
            "Final_Accuracy": res["final_accuracy"],
            "Final_AUC": res["final_auc"],
            "Worst_Hospital_AUC": res["worst_auc"],
        }
        dp_results.append(dp_entry)
        print(f"      Result: Acc={res['final_accuracy']:.4f} | AUC={res['final_auc']:.4f}")

    # Save DP CSV
    dp_csv = RESULTS_DIR / "phase4_dp_sweep.csv"
    pd.DataFrame(dp_results).to_csv(dp_csv, index=False)
    print(f"\n[OK] Differential Privacy sweep saved -> {dp_csv}")

    # ─────────────────────────────────────────────────────────────
    # PART 3: Communication Cost Accounting
    # ─────────────────────────────────────────────────────────────
    print("\n[3/3] Calculating Communication Cost Accounting...")
    eval_model = Net()
    sample_params = get_parameters(eval_model)
    comm_stats = compute_federation_comm_cost(
        num_clients=6,
        num_rounds=rounds,
        parameters=sample_params,
        centralized_dataset_bytes=490_000,
    )
    comm_csv = RESULTS_DIR / "phase4_comm_cost.csv"
    pd.DataFrame([comm_stats]).to_csv(comm_csv, index=False)
    print(f"[OK] Communication cost breakdown saved -> {comm_csv}")

    # ─────────────────────────────────────────────────────────────
    # Print Summary Tables
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 85)
    print("  TABLE 4: ATTACK x DEFENSE BENCHMARK MATRIX (FINAL ROC-AUC)")
    print("=" * 85)
    print(f"  {'Attack Scenario':<35} | {'FedAvg':<8} | {'Trimmed Mean':<14} | {'Median':<8} | {'Multi-Krum'}")
    print("-" * 85)

    df_mat = pd.DataFrame(matrix_results)
    for att_id, att_name in attacks:
        row_str = f"  {att_name:<35} | "
        for strat_id, _ in strategies:
            match = df_mat[(df_mat["Attack_Type"] == att_id) & (df_mat["Strategy"] == strat_id)]
            auc_val = match["Final_AUC"].values[0] if len(match) > 0 else 0.0
            if strat_id == "fedavg":
                row_str += f"{auc_val:<8.4f} | "
            elif strat_id == "trimmed_mean":
                row_str += f"{auc_val:<14.4f} | "
            elif strat_id == "median":
                row_str += f"{auc_val:<8.4f} | "
            else:
                row_str += f"{auc_val:.4f}"
        print(row_str)
    print("=" * 85)

    print("\n" + "=" * 80)
    print("  TABLE 5: DIFFERENTIAL PRIVACY (DP) UTILITY TRADEOFF")
    print("=" * 80)
    print(f"  {'Noise sigma':<11} | {'Epsilon (eps)':<15} | {'Privacy Level':<22} | {'AUC':<7} | {'Accuracy'}")
    print("-" * 80)
    for r in dp_results:
        eps_str = f"{r['Epsilon']:.2f}" if not np.isinf(r["Epsilon"]) else "inf (None)"
        print(
            f"  {r['Noise_Multiplier']:<11.4f} | "
            f"{eps_str:<15} | "
            f"{r['Privacy_Regime']:<22} | "
            f"{r['Final_AUC']:<7.4f} | "
            f"{r['Final_Accuracy']:.4f}"
        )
    print("=" * 80 + "\n")


if __name__ == "__main__":
    args = parse_args()
    run_all_phase4_experiments(
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )

"""
run_federated.py - Federated Learning Simulation Runner (FedAvg).

Executes privacy-preserving federated training across 6 non-IID hospital nodes
using the Flower FedAvg protocol.

Workflow:
1. Initialize 6 autonomous hospital clients with data isolation (Hospital 1 to 6).
2. For each round:
   - Server broadcasts global parameters to all hospitals.
   - Each hospital executes local gradient descent on its private data.
   - Server aggregates updates with sample-weighted averaging.
   - Server performs global evaluation on the pooled held-out test split.
   - Each hospital evaluates the global model locally to track per-node utility.
3. Logs round-by-round trajectory to CSV.
4. Outputs comparison table: Centralized vs Local-Only vs Federated.

Usage:
    python run_federated.py --rounds 20 --local-epochs 2 --lr 0.001
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from fedcare.client_app import FlowerClient, get_parameters, set_parameters
from fedcare.server_app import create_fedavg_strategy, get_initial_parameters
from fedcare.strategy.fedavg_weighted import FedAvgWeighted
from fedcare.task import Net, evaluate, load_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("fedcare.run_federated")

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="FedCare - Federated Learning with FedAvg across 6 Hospitals."
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=20,
        help="Number of federated communication rounds (default: 20).",
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=2,
        help="Number of local training epochs per round per client (default: 2).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for local client training (default: 0.001).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training and evaluation (default: 32).",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=6,
        help="Number of hospital clients (default: 6).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=True,
        help="Automatically generate convergence plot after simulation (default: True).",
    )
    return parser.parse_args()


def run_federation(
    num_rounds: int = 20,
    local_epochs: int = 2,
    lr: float = 0.001,
    batch_size: int = 32,
    num_clients: int = 6,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Execute the federated learning simulation.

    Parameters
    ----------
    num_rounds : int
        Total federation rounds.
    local_epochs : int
        Local training epochs per client per round.
    lr : float
        Client learning rate.
    batch_size : int
        Batch size.
    num_clients : int
        Number of participating hospitals.

    Returns
    -------
    tuple[dict[str, Any], list[dict[str, Any]]]
        Final summary metrics and round-by-round metrics list.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 64)
    print("  FEDCARE - FEDERATED LEARNING CORE (FLOWER + FEDAVG)")
    print("=" * 64)
    print(f"  Participating Hospitals : {num_clients}")
    print(f"  Federation Rounds       : {num_rounds}")
    print(f"  Local Epochs / Round    : {local_epochs}")
    print(f"  Client Learning Rate    : {lr}")
    print(f"  Batch Size              : {batch_size}")
    print("=" * 64 + "\n")

    # 1. Initialize hospitals (data isolation strictly preserved)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Initializing hospital nodes...")
    clients: list[FlowerClient] = []
    for hospital_id in range(1, num_clients + 1):
        client = FlowerClient(
            partition_id=hospital_id,
            batch_size=batch_size,
            device=device,
        )
        clients.append(client)
        print(
            f"  [Hospital {hospital_id}] Train samples: {client.num_train_samples}, "
            f"Test samples: {client.num_test_samples}"
        )

    # 2. Initialize strategy with server-side evaluation
    strategy = create_fedavg_strategy(
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        lr=lr,
        min_clients=num_clients,
        checkpoint_dir=CHECKPOINT_DIR,
    )

    # 3. Global model state
    global_parameters = strategy.initial_parameters
    if global_parameters is None:
        global_parameters = get_initial_parameters()

    # Load pooled test loader for central server evaluation
    _, server_test_loader, _ = load_data(partition_id=None, batch_size=batch_size)
    eval_model = Net().to(device)

    # Round trajectory container
    history: list[dict[str, Any]] = []

    print("\nStarting Federated Training Loop...")
    print("-" * 64)
    print(
        f"  {'Round':<7} | {'Train Loss':<11} | {'Test Loss':<10} | "
        f"{'Accuracy':<9} | {'AUC':<7} | {'Hosp Avg AUC':<12}"
    )
    print("-" * 64)

    for server_round in range(1, num_rounds + 1):
        # ── A. Client Local Training (Fit) ──
        fit_config = {
            "server_round": server_round,
            "local_epochs": local_epochs,
            "lr": lr,
        }
        ndarrays_global = parameters_to_ndarrays(global_parameters)

        fit_results: list[tuple[Any, int, dict[str, Any]]] = []
        train_losses: list[float] = []

        for client in clients:
            updated_ndarrays, num_samples, metrics = client.fit(
                parameters=ndarrays_global,
                config=fit_config,
            )
            fit_results.append((updated_ndarrays, num_samples, metrics))
            train_losses.append(float(metrics.get("train_loss", 0.0)))

        avg_train_loss = float(np.mean(train_losses))

        # ── B. Server Aggregation (FedAvg Weighted) ──
        total_fit_samples = sum(n for _, n, _ in fit_results)
        aggregated_ndarrays: list[np.ndarray] = [
            np.zeros_like(w) for w in ndarrays_global
        ]

        for client_weights, num_samples, _ in fit_results:
            weight_fraction = num_samples / total_fit_samples
            for i, layer in enumerate(client_weights):
                aggregated_ndarrays[i] += layer * weight_fraction

        global_parameters = ndarrays_to_parameters(aggregated_ndarrays)

        # ── C. Global Evaluation on Pooled Held-out Test Set ──
        set_parameters(eval_model, aggregated_ndarrays)
        eval_metrics = evaluate(
            eval_model,
            server_test_loader,
            device=device,
        )
        server_loss = float(eval_metrics["loss"])
        server_acc = float(eval_metrics["accuracy"])
        server_auc = float(eval_metrics["auc"])

        # ── D. Local Evaluation on Each Hospital's Own Test Split ──
        hosp_aucs: list[float] = []
        hosp_accs: list[float] = []
        for client in clients:
            h_loss, h_samples, h_metrics = client.evaluate(
                parameters=aggregated_ndarrays,
                config={"server_round": server_round},
            )
            hosp_aucs.append(float(h_metrics["auc"]))
            hosp_accs.append(float(h_metrics["accuracy"]))

        avg_hosp_auc = float(np.mean(hosp_aucs))
        avg_hosp_acc = float(np.mean(hosp_accs))

        # Record round entry
        round_data: dict[str, Any] = {
            "round": server_round,
            "train_loss": round(avg_train_loss, 4),
            "test_loss": round(server_loss, 4),
            "accuracy": round(server_acc, 4),
            "auc": round(server_auc, 4),
            "hosp_avg_accuracy": round(avg_hosp_acc, 4),
            "hosp_avg_auc": round(avg_hosp_auc, 4),
        }
        # Add individual hospital AUCs for deep analysis
        for idx, h_auc in enumerate(hosp_aucs, start=1):
            round_data[f"hosp_{idx}_auc"] = round(h_auc, 4)

        history.append(round_data)

        print(
            f"  Round {server_round:>2} | "
            f"{avg_train_loss:>10.4f}  | "
            f"{server_loss:>9.4f}  | "
            f"{server_acc:>8.4f}  | "
            f"{server_auc:>6.4f} | "
            f"{avg_hosp_auc:>12.4f}"
        )

    print("-" * 64)

    # 4. Save round history to CSV
    csv_path = RESULTS_DIR / "rounds_fedavg.csv"
    if history:
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
        print(f"\n[OK] Round trajectory saved -> {csv_path}")

    # 5. Best results extraction
    best_round = max(history, key=lambda x: x["auc"])
    final_round = history[-1]

    # Save final model parameters
    final_model_path = CHECKPOINT_DIR / "final_fedavg_model.pt"
    torch.save(eval_model.state_dict(), final_model_path)
    print(f"[OK] Final global model saved -> {final_model_path}")

    # 6. Print Comparison Table
    print("\n" + "=" * 64)
    print("  FEDCARE PHASE 2 RESULTS COMPARISON")
    print("=" * 64)
    print(f"  {'Method':<25} | {'Accuracy':<10} | {'AUC':<10} | {'Privacy'}")
    print("-" * 64)
    print(f"  {'Centralized (Pooled)':<25} | {'0.8083':<10} | {'0.8480':<10} | Data Shared")
    print(f"  {'Local-Only (Average)':<25} | {'0.8071':<10} | {'0.8121':<10} | Isolated (No FL)")
    print(
        f"  {'FedAvg (Final Round)':<25} | "
        f"{final_round['accuracy']:<10.4f} | "
        f"{final_round['auc']:<10.4f} | "
        f"Privacy Preserved"
    )
    print(
        f"  {'FedAvg (Best Peak)':<25} | "
        f"{best_round['accuracy']:<10.4f} | "
        f"{best_round['auc']:<10.4f} | "
        f"Privacy Preserved (Round {best_round['round']})"
    )
    print("=" * 64)

    gap = (final_round["auc"] - 0.8121) / (0.8480 - 0.8121) * 100.0
    print(f"\n>> FedAvg recovered {gap:.1f}% of the Centralized-Local AUC gap")
    print("   WITHOUT sharing a single patient record between hospitals!\n")

    summary = {
        "final_accuracy": final_round["accuracy"],
        "final_auc": final_round["auc"],
        "best_accuracy": best_round["accuracy"],
        "best_auc": best_round["auc"],
        "best_round": best_round["round"],
    }
    return summary, history


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    summary, history = run_federation(
        num_rounds=args.rounds,
        local_epochs=args.local_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        num_clients=args.num_clients,
    )

    if args.plot:
        try:
            from scripts.plot_phase2 import plot_fedavg_convergence

            plot_fedavg_convergence()
        except ImportError:
            logger.warning("scripts.plot_phase2 not found. Skipping plot generation.")


if __name__ == "__main__":
    main()

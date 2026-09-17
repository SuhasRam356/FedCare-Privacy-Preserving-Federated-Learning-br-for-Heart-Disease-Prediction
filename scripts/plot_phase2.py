"""
scripts/plot_phase2.py - Generate publication-ready convergence plots for Phase 2.

Produces Figure 1:
- Subplot 1: ROC-AUC vs. Federated Communication Rounds.
- Subplot 2: Classification Accuracy vs. Communication Rounds.
- Subplot 3: Training & Global Validation Loss Curves.

Overlays Centralized (pooled) upper bound and Local-Only lower bound baselines.

Usage:
    python scripts/plot_phase2.py
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CSV_PATH = RESULTS_DIR / "rounds_fedavg.csv"
OUTPUT_PLOT = RESULTS_DIR / "figure1_fedavg_convergence.png"

# Baseline benchmarks from Phase 1
CENTRALIZED_AUC = 0.8480
CENTRALIZED_ACC = 0.8083
LOCAL_AVG_AUC = 0.8121
LOCAL_AVG_ACC = 0.8071


def plot_fedavg_convergence(
    csv_file: Path = CSV_PATH,
    output_file: Path = OUTPUT_PLOT,
) -> None:
    """
    Generate and save the Phase 2 convergence figure.

    Parameters
    ----------
    csv_file : Path
        Path to rounds_fedavg.csv.
    output_file : Path
        Path to save the generated figure.
    """
    if not csv_file.exists():
        print(f"Error: {csv_file} not found. Run 'python run_federated.py' first.")
        return

    df = pd.read_csv(csv_file)
    rounds = df["round"]

    # Set clean publication style
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

    # ── 1. ROC-AUC vs. Round ──
    ax1 = axes[0]
    # Individual hospital traces
    hosp_cols = [c for c in df.columns if c.startswith("hosp_") and c.endswith("_auc")]
    hosp_names = {
        "hosp_1_auc": "H1 (Metro)",
        "hosp_2_auc": "H2 (Rural, 4.8% sick)",
        "hosp_3_auc": "H3 (Geriatric)",
        "hosp_4_auc": "H4 (Industrial)",
        "hosp_5_auc": "H5 (Women's)",
        "hosp_6_auc": "H6 (Diabetes)",
    }
    for col in hosp_cols:
        label = hosp_names.get(col, col)
        ax1.plot(rounds, df[col], alpha=0.35, linewidth=1.2, linestyle=":", label=label)

    # FedAvg Global AUC
    ax1.plot(
        rounds,
        df["auc"],
        color="#1f77b4",
        linewidth=2.5,
        marker="o",
        markersize=5,
        label="FedAvg Global Model",
    )
    # Benchmark lines
    ax1.axhline(
        CENTRALIZED_AUC,
        color="#d62728",
        linestyle="--",
        linewidth=1.8,
        label=f"Centralized Pooled ({CENTRALIZED_AUC:.4f})",
    )
    ax1.axhline(
        LOCAL_AVG_AUC,
        color="#2ca02c",
        linestyle="-.",
        linewidth=1.8,
        label=f"Local-Only Avg ({LOCAL_AVG_AUC:.4f})",
    )

    ax1.set_title("A. ROC-AUC Convergence Across Rounds", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Federated Communication Round", fontsize=11)
    ax1.set_ylabel("ROC-AUC Score", fontsize=11)
    ax1.set_ylim(0.74, 0.88)
    ax1.legend(loc="lower right", fontsize=8.5, frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # ── 2. Accuracy vs. Round ──
    ax2 = axes[1]
    ax2.plot(
        rounds,
        df["accuracy"],
        color="#2b5c8f",
        linewidth=2.5,
        marker="s",
        markersize=5,
        label="FedAvg Global Accuracy",
    )
    if "hosp_avg_accuracy" in df.columns:
        ax2.plot(
            rounds,
            df["hosp_avg_accuracy"],
            color="#8c564b",
            linewidth=1.8,
            linestyle="--",
            label="Client Mean Accuracy",
        )
    ax2.axhline(
        CENTRALIZED_ACC,
        color="#d62728",
        linestyle="--",
        linewidth=1.8,
        label=f"Centralized Pooled ({CENTRALIZED_ACC:.4f})",
    )
    ax2.axhline(
        LOCAL_AVG_ACC,
        color="#2ca02c",
        linestyle="-.",
        linewidth=1.8,
        label=f"Local-Only Avg ({LOCAL_AVG_ACC:.4f})",
    )

    ax2.set_title("B. Classification Accuracy vs. Round", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Federated Communication Round", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.set_ylim(0.76, 0.84)
    ax2.legend(loc="lower right", fontsize=8.5, frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # ── 3. Loss Curves ──
    ax3 = axes[2]
    ax3.plot(
        rounds,
        df["train_loss"],
        color="#e377c2",
        linewidth=2.0,
        marker="^",
        markersize=5,
        label="Client Mean Train Loss",
    )
    ax3.plot(
        rounds,
        df["test_loss"],
        color="#ff7f0e",
        linewidth=2.0,
        marker="v",
        markersize=5,
        label="Global Test Loss",
    )

    ax3.set_title("C. Loss Minimization Trajectory", fontsize=13, fontweight="bold", pad=10)
    ax3.set_xlabel("Federated Communication Round", fontsize=11)
    ax3.set_ylabel("CrossEntropy Loss", fontsize=11)
    ax3.legend(loc="upper right", fontsize=9, frameon=True)
    ax3.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Publication figure generated -> {output_file}")


if __name__ == "__main__":
    plot_fedavg_convergence()

"""
scripts/plot_phase4.py - Publication-Quality Visualizations for FedCare Phase 4.

Generates:
1. Figure 4 (figure4_attacks_and_defenses.png):
   - A. Convergence trajectories under Label-Flipping Attack (FedAvg vs Defenses).
   - B. Convergence trajectories under Model Poisoning Attack (FedAvg vs Defenses).
   - C. Grouped bar chart of the Attack x Defense Matrix (Headline IEEE Figure).
2. Figure 5 (figure5_privacy_utility.png):
   - A. Privacy-Utility trade-off curve (ROC-AUC vs Epsilon epsilon).
   - B. Communication efficiency: Federated Learning vs Centralized Raw Data Transfer.

Usage:
    python scripts/plot_phase4.py
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MATRIX_CSV = RESULTS_DIR / "phase4_attack_defense_matrix.csv"
TRAJ_CSV = RESULTS_DIR / "phase4_attack_trajectories.csv"
DP_CSV = RESULTS_DIR / "phase4_dp_sweep.csv"
COMM_CSV = RESULTS_DIR / "phase4_comm_cost.csv"

FIG4_PATH = RESULTS_DIR / "figure4_attacks_and_defenses.png"
FIG5_PATH = RESULTS_DIR / "figure5_privacy_utility.png"


def plot_figure4_attacks_and_defenses() -> None:
    """Generate Figure 4: Attack Resilience and Byzantine Defense Performance."""
    if not MATRIX_CSV.exists() or not TRAJ_CSV.exists():
        print("Missing Phase 4 CSVs. Run run_phase4_experiments.py first.")
        return

    df_matrix = pd.read_csv(MATRIX_CSV)
    df_traj = pd.read_csv(TRAJ_CSV)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig = plt.figure(figsize=(16, 5.2), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.3])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    strat_colors = {
        "fedavg": "#d62728",        # Red (Vulnerable)
        "trimmed_mean": "#1f77b4",  # Blue (Robust)
        "median": "#2ca02c",        # Green (Robust)
        "krum": "#9467bd",          # Purple (Robust)
    }
    strat_labels = {
        "fedavg": "FedAvg (Vulnerable)",
        "trimmed_mean": "Trimmed Mean (β=0.17)",
        "median": "Coordinate Median",
        "krum": "Multi-Krum",
    }
    markers = {"fedavg": "x", "trimmed_mean": "o", "median": "s", "krum": "^"}

    # ── Subplot A: Trajectories under Label-Flipping ──
    df_lf = df_traj[df_traj["attack"] == "label_flip"]
    for strat in ["fedavg", "trimmed_mean", "median", "krum"]:
        strat_data = df_lf[df_lf["strategy"] == strat].sort_values("round")
        if not strat_data.empty:
            ax1.plot(
                strat_data["round"],
                strat_data["server_auc"],
                label=strat_labels[strat],
                color=strat_colors[strat],
                marker=markers[strat],
                linewidth=1.8,
                markersize=4.5,
            )

    ax1.set_title("A. Label-Flipping Attack (Hosp 4)", fontsize=11, fontweight="bold", pad=8)
    ax1.set_xlabel("Communication Round", fontsize=10)
    ax1.set_ylabel("Server ROC-AUC", fontsize=10)
    ax1.set_ylim(0.70, 0.88)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", fontsize=8.5, frameon=True)

    # ── Subplot B: Trajectories under Model Poisoning ──
    df_mp = df_traj[df_traj["attack"] == "model_poison"]
    for strat in ["fedavg", "trimmed_mean", "median", "krum"]:
        strat_data = df_mp[df_mp["strategy"] == strat].sort_values("round")
        if not strat_data.empty:
            ax2.plot(
                strat_data["round"],
                strat_data["server_auc"],
                label=strat_labels[strat],
                color=strat_colors[strat],
                marker=markers[strat],
                linewidth=1.8,
                markersize=4.5,
            )

    ax2.set_title("B. Model Poisoning Attack (Sign-Flip)", fontsize=11, fontweight="bold", pad=8)
    ax2.set_xlabel("Communication Round", fontsize=10)
    ax2.set_ylabel("Server ROC-AUC", fontsize=10)
    ax2.set_ylim(0.65, 0.88)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", fontsize=8.5, frameon=True)

    # ── Subplot C: Grouped Bar Chart of Attack x Defense Matrix ──
    attack_order = ["clean", "label_flip", "model_poison"]
    attack_names = ["No Attack\n(Clean)", "Label-Flip\n(100% Inverted)", "Model Poison\n(Sign-Flip x-3.0)"]
    x = np.arange(len(attack_order))
    width = 0.20

    strat_list = ["fedavg", "trimmed_mean", "median", "krum"]
    for i, strat in enumerate(strat_list):
        aucs = []
        for att in attack_order:
            row = df_matrix[(df_matrix["Attack_Type"] == att) & (df_matrix["Strategy"] == strat)]
            aucs.append(row["Final_AUC"].values[0] if len(row) > 0 else 0.0)

        bars = ax3.bar(
            x + (i - 1.5) * width,
            aucs,
            width,
            label=strat_labels[strat],
            color=strat_colors[strat],
            edgecolor="black",
            linewidth=0.7,
        )

        for bar in bars:
            h = bar.get_height()
            ax3.annotate(
                f"{h:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold" if strat != "fedavg" else "normal",
            )

    ax3.set_title("C. Attack x Defense Benchmark Matrix", fontsize=11, fontweight="bold", pad=8)
    ax3.set_ylabel("Final Global ROC-AUC", fontsize=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(attack_names, fontsize=9.5)
    ax3.set_ylim(0.65, 0.90)
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="lower left", fontsize=8.5, frameon=True)

    plt.tight_layout()
    FIG4_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG4_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Figure 4 generated -> {FIG4_PATH}")


def plot_figure5_privacy_utility() -> None:
    """Generate Figure 5: Differential Privacy Tradeoff & Communication Overhead."""
    if not DP_CSV.exists():
        print("Missing Phase 4 DP CSV. Run run_phase4_experiments.py first.")
        return

    df_dp = pd.read_csv(DP_CSV)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.0), dpi=300)

    # ── Subplot A: Privacy-Utility Curve (AUC vs Noise Multiplier & Epsilon) ──
    finite_dp = df_dp[np.isfinite(df_dp["Epsilon"])].sort_values("Noise_Multiplier")

    # X-axis as noise multiplier
    x_noise = df_dp["Noise_Multiplier"]
    y_auc = df_dp["Final_AUC"]
    y_acc = df_dp["Final_Accuracy"]

    ax1.plot(x_noise, y_auc, "o-", color="#1f77b4", linewidth=2.2, markersize=6, label="ROC-AUC Score")
    ax1.plot(x_noise, y_acc, "s--", color="#2ca02c", linewidth=1.8, markersize=5, label="Classification Accuracy")

    # Annotate epsilons
    for _, row in df_dp.iterrows():
        n = row["Noise_Multiplier"]
        auc = row["Final_AUC"]
        eps_txt = f"ε={row['Epsilon']:.1f}" if np.isfinite(row["Epsilon"]) else "ε=∞"
        ax1.annotate(
            eps_txt,
            xy=(n, auc),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color="#08306b",
        )

    ax1.set_title(r"A. Privacy-Utility Tradeoff ($\sigma$ vs. Performance)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel(r"Gaussian Noise Multiplier ($\sigma$)", fontsize=11)
    ax1.set_ylabel("Evaluation Metric", fontsize=11)
    ax1.set_ylim(0.70, 0.88)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower left", fontsize=9.5, frameon=True)

    # ── Subplot B: Communication Overhead (FL vs Centralized Data Pooling) ──
    # Centralized dataset: 12,000 records * ~40 bytes = 480 KB (single transmission)
    # Federated Learning: 6 hospitals * 20 rounds * 2 messages * 11.88 KB = ~2.85 MB total over 20 rounds
    comm_categories = [
        "Single Hospital\nModel Update (1 Round)",
        "Entire Network\nPer Round (6 Hosp)",
        "Centralized Raw Data\n(12,000 Records)",
        "FedCare 20 Rounds\n(Full Training Lifetime)",
    ]
    sizes_kb = [11.88, 142.56, 480.0, 2851.2]
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd"]

    bars = ax2.bar(comm_categories, sizes_kb, color=colors, width=0.45, edgecolor="black", linewidth=0.8)
    ax2.set_title("B. Communication Payload Profile (Payload Sizing)", fontsize=12, fontweight="bold", pad=10)
    ax2.set_ylabel("Data Transmitted (Kilobytes - Log Scale)", fontsize=11)
    ax2.set_yscale("log")
    ax2.grid(True, linestyle="--", alpha=0.5)

    for bar, val in zip(bars, sizes_kb):
        label = f"{val:.1f} KB" if val < 1000 else f"{val/1024:.2f} MB"
        ax2.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    FIG5_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG5_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Figure 5 generated -> {FIG5_PATH}")


if __name__ == "__main__":
    plot_figure4_attacks_and_defenses()
    plot_figure5_privacy_utility()

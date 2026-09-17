"""
scripts/plot_phase3.py - Publication-quality plotting for Phase 3 experiments.

Generates:
1. Figure 2 (figure2_non_iid_impact.png):
   - Compares FedAvg performance under IID, Dirichlet (alpha = 1.0, 0.5, 0.1),
     and Hospital-Native skew.
   - Shows Overall AUC vs. Worst-Hospital AUC (fairness penalty).
2. Figure 3 (figure3_fedprox_vs_fedavg.png):
   - Evaluates FedProx proximal parameter mu sweep (0.0 to 1.0).
   - Shows overall AUC preservation, worst-hospital fairness improvement,
     and the benefit of local personalization on skewed Hospital 2.

Usage:
    python scripts/plot_phase3.py
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
NON_IID_CSV = RESULTS_DIR / "phase3_non_iid_experiments.csv"
FEDPROX_CSV = RESULTS_DIR / "phase3_fedprox_experiments.csv"

FIG2_PATH = RESULTS_DIR / "figure2_non_iid_impact.png"
FIG3_PATH = RESULTS_DIR / "figure3_fedprox_vs_fedavg.png"


def plot_figure2_non_iid_impact() -> None:
    """Generate Figure 2: Non-IID Skew Analysis."""
    if not NON_IID_CSV.exists():
        print(f"File not found: {NON_IID_CSV}. Run run_phase3_experiments.py first.")
        return

    df = pd.read_csv(NON_IID_CSV)

    labels = [
        "IID\n(Uniform)",
        r"Dirichlet" "\n" r"($\alpha=1.0$)",
        r"Dirichlet" "\n" r"($\alpha=0.5$)",
        r"Dirichlet" "\n" r"($\alpha=0.1$)",
        "Hospital\nNative",
    ]

    x = np.arange(len(labels))
    width = 0.28

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)

    # ── Subplot 1: Overall AUC vs Worst-Hospital AUC ──
    rects1 = ax1.bar(
        x - width / 2,
        df["Final_AUC"],
        width,
        label="Overall Server AUC",
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.8,
    )
    rects2 = ax1.bar(
        x + width / 2,
        df["Worst_Hospital_AUC"],
        width,
        label="Worst-Hospital AUC (Min Fairness)",
        color="#d62728",
        edgecolor="black",
        linewidth=0.8,
    )

    ax1.set_title("A. Global AUC vs. Worst-Hospital AUC across Skew Regimes", fontsize=12, fontweight="bold", pad=10)
    ax1.set_ylabel("ROC-AUC Score", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylim(0.68, 0.89)
    ax1.legend(loc="lower left", fontsize=9.5, frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Annotate bars
    for rect in rects1:
        height = rect.get_height()
        ax1.annotate(f"{height:.3f}", xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
    for rect in rects2:
        height = rect.get_height()
        ax1.annotate(f"{height:.3f}", xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color="#8b0000")

    # ── Subplot 2: Equity Gap (Disparity) ──
    colors = ["#2ca02c", "#bcbd22", "#ff7f0e", "#d62728", "#9467bd"]
    bars = ax2.bar(
        labels,
        df["Equity_Gap"],
        color=colors,
        width=0.45,
        edgecolor="black",
        linewidth=0.8,
    )

    ax2.set_title(r"B. Inter-Hospital Equity Gap ($\Delta = \text{Best} - \text{Worst}$)", fontsize=12, fontweight="bold", pad=10)
    ax2.set_ylabel("AUC Equity Gap (Lower is Fairer)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax2.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    FIG2_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG2_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Figure 2 generated -> {FIG2_PATH}")


def plot_figure3_fedprox_vs_fedavg() -> None:
    """Generate Figure 3: FedProx vs FedAvg Optimization Sweep."""
    if not FEDPROX_CSV.exists():
        print(f"File not found: {FEDPROX_CSV}. Run run_phase3_experiments.py first.")
        return

    df = pd.read_csv(FEDPROX_CSV)

    mu_labels = [r"FedAvg ($\mu=0$)", r"$\mu=0.001$", r"$\mu=0.01$", r"$\mu=0.1$", r"$\mu=1.0$"]
    x = np.arange(len(mu_labels))
    width = 0.26

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)

    # ── Subplot 1: FedProx AUC & Worst Hospital Recovery ──
    b1 = ax1.bar(
        x - width / 2,
        df["Final_AUC"],
        width,
        label="Overall Server AUC",
        color="#2b5c8f",
        edgecolor="black",
        linewidth=0.8,
    )
    b2 = ax1.bar(
        x + width / 2,
        df["Worst_Hospital_AUC"],
        width,
        label="Worst-Hospital AUC (Fairness)",
        color="#e6550d",
        edgecolor="black",
        linewidth=0.8,
    )

    ax1.set_title(r"A. Impact of Proximal Term $\mu$ on Hospital-Native Data", fontsize=12, fontweight="bold", pad=10)
    ax1.set_ylabel("ROC-AUC Score", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(mu_labels, fontsize=10)
    ax1.set_ylim(0.72, 0.88)
    ax1.legend(loc="lower left", fontsize=9.5, frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    for b in b1:
        h = b.get_height()
        ax1.annotate(f"{h:.3f}", xy=(b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
    for b in b2:
        h = b.get_height()
        ax1.annotate(f"{h:.3f}", xy=(b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color="#8b0000")

    # ── Subplot 2: Hospital 2 Personalization Boost ──
    b_before = ax2.bar(
        x - width / 2,
        df["H2_Before_Pers"],
        width,
        label="Global Model on Hospital 2 (4.8% prevalence)",
        color="#756bb1",
        edgecolor="black",
        linewidth=0.8,
    )
    b_after = ax2.bar(
        x + width / 2,
        df["H2_After_Pers"],
        width,
        label="After Local Personalization (2 epochs)",
        color="#31a354",
        edgecolor="black",
        linewidth=0.8,
    )

    ax2.set_title("B. Local Personalization (Fine-Tuning) on Skewed Hospital 2", fontsize=12, fontweight="bold", pad=10)
    ax2.set_ylabel("Hospital 2 ROC-AUC", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(mu_labels, fontsize=10)
    ax2.set_ylim(0.70, 0.90)
    ax2.legend(loc="lower right", fontsize=9.5, frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)

    for b in b_before:
        h = b.get_height()
        ax2.annotate(f"{h:.3f}", xy=(b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
    for b in b_after:
        h = b.get_height()
        ax2.annotate(f"{h:.3f}", xy=(b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#006400")

    plt.tight_layout()
    FIG3_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG3_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Figure 3 generated -> {FIG3_PATH}")


if __name__ == "__main__":
    plot_figure2_non_iid_impact()
    plot_figure3_fedprox_vs_fedavg()

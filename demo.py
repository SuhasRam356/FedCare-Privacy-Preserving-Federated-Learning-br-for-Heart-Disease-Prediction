"""
FedCare Demo Launcher
=====================
One-command script to launch the FedCare interactive dashboard
for the M.Tech viva presentation.

Usage:
    python demo.py
    
This script:
    1. Verifies all dependencies are installed
    2. Checks that training data and results exist
    3. Ensures a trained model checkpoint is available
    4. Launches the Streamlit dashboard on localhost
"""

from __future__ import annotations

import os
import sys
import subprocess
import time
from pathlib import Path

# ── Project Paths ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "heart"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DASHBOARD_PATH = PROJECT_ROOT / "app" / "dashboard.py"


def check_banner():
    """Display the FedCare startup banner."""
    banner = r"""
    ================================================================
     ___        _ ___                
    | __|___ __| / __|__ _ _ _ ___   
    | _/ -_) _` | (__/ _` | '_/ -_)  
    |_|\___\__,_|\___\__,_|_| \___|  
                                     
    Privacy-Preserving Federated Learning
    for Heart Disease Prediction
    ================================================================
    M.Tech Viva Demonstration Dashboard
    ================================================================
    """
    print(banner)


def check_dependencies():
    """Verify all required Python packages are installed."""
    print("[1/5] Checking dependencies...")
    required = {
        "streamlit": "streamlit",
        "plotly": "plotly",
        "torch": "torch",
        "flwr": "flwr",
        "pandas": "pandas",
        "numpy": "numpy",
        "sklearn": "scikit-learn",
    }

    missing = []
    for import_name, pip_name in required.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)

    if missing:
        print(f"  -> Installing missing packages: {', '.join(missing)}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet"] + missing,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("  -> All dependencies installed successfully.")
    else:
        print("  -> All dependencies are available.")


def check_data():
    """Verify training data files exist."""
    print("[2/5] Checking training data...")
    required_files = [
        DATA_DIR / "combined.csv",
    ] + [DATA_DIR / f"hospital_{i}.csv" for i in range(1, 7)]

    missing = [f for f in required_files if not f.exists()]
    if missing:
        print("  !! ERROR: Missing data files:")
        for f in missing:
            print(f"     - {f}")
        print("  -> Please ensure all hospital CSV files are in data/heart/")
        sys.exit(1)
    else:
        import pandas as pd
        total = 0
        for i in range(1, 7):
            df = pd.read_csv(DATA_DIR / f"hospital_{i}.csv")
            total += len(df)
        print(f"  -> Found 6 hospital datasets ({total:,} total patients).")


def check_results():
    """Check if experiment results exist."""
    print("[3/5] Checking experiment results...")
    result_files = [
        "rounds_fedavg.csv",
        "phase3_non_iid_experiments.csv",
        "phase3_fedprox_experiments.csv",
        "phase4_attack_defense_matrix.csv",
        "phase4_dp_sweep.csv",
        "phase4_comm_cost.csv",
    ]

    found = 0
    for f in result_files:
        if (RESULTS_DIR / f).exists():
            found += 1

    if found == 0:
        print("  !! WARNING: No experiment results found.")
        print("  -> Dashboard will have limited visualizations.")
        print("  -> Run training scripts first for full experience.")
    else:
        print(f"  -> Found {found}/{len(result_files)} experiment result files.")


def check_figures():
    """Check if pre-generated research figures exist."""
    print("[4/5] Checking research figures...")
    figure_files = [
        "figure1_fedavg_convergence.png",
        "figure2_non_iid_impact.png",
        "figure3_fedprox_vs_fedavg.png",
        "figure4_attacks_and_defenses.png",
        "figure5_privacy_utility.png",
    ]

    found = sum(1 for f in figure_files if (RESULTS_DIR / f).exists())
    print(f"  -> Found {found}/{len(figure_files)} research figures.")


def ensure_checkpoint():
    """Ensure a model checkpoint exists for the risk calculator."""
    print("[5/5] Checking model checkpoint...")
    checkpoint_path = CHECKPOINT_DIR / "final_fedavg_model.pt"

    if checkpoint_path.exists():
        print("  -> Global model checkpoint found.")
        return

    # Train a quick centralized model as fallback
    print("  -> No checkpoint found. Training a quick centralized model...")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        sys.path.insert(0, str(PROJECT_ROOT))
        from fedcare.task import Net, load_data, train

        model = Net()
        train_loader, test_loader, _ = load_data(partition_id=None)

        # Train for 15 epochs
        for epoch in range(15):
            loss = train(model, train_loader, epochs=1, lr=0.001)
            if (epoch + 1) % 5 == 0:
                print(f"     Epoch {epoch + 1}/15 - Loss: {loss:.4f}")

        # Save checkpoint
        torch.save(model.state_dict(), checkpoint_path)
        print(f"  -> Model saved to {checkpoint_path}")

        # Quick evaluation
        from fedcare.task import evaluate
        metrics = evaluate(model, test_loader)
        print(f"  -> Quick eval: Accuracy={metrics['accuracy']:.4f}, AUC={metrics['auc']:.4f}")

    except Exception as e:
        print(f"  !! Warning: Could not train model ({e}).")
        print("  -> Risk calculator will train on-the-fly when first used.")


def launch_dashboard():
    """Launch the Streamlit dashboard."""
    print("\n" + "=" * 64)
    print("  Launching FedCare Dashboard...")
    print("=" * 64)
    print(f"\n  Dashboard: {DASHBOARD_PATH}")
    print("  URL: http://localhost:8501")
    print("\n  Press Ctrl+C to stop the dashboard.\n")

    # Configure Streamlit settings
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ["STREAMLIT_THEME_BASE"] = "dark"

    # Launch Streamlit
    subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run",
            str(DASHBOARD_PATH),
            "--server.port=8501",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false",
            "--theme.base=dark",
            "--theme.primaryColor=#6366f1",
            "--theme.backgroundColor=#0f172a",
            "--theme.secondaryBackgroundColor=#1e293b",
            "--theme.textColor=#f1f5f9",
        ],
        cwd=str(PROJECT_ROOT),
    )


def main():
    """Main demo launcher entry point."""
    check_banner()
    check_dependencies()
    check_data()
    check_results()
    check_figures()
    ensure_checkpoint()
    launch_dashboard()


if __name__ == "__main__":
    main()

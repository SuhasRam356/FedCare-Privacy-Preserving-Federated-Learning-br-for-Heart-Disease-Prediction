"""
FedCare Data Loaders
====================
Cached data loading functions for all experiment results and hospital data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# ── Path Setup ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "heart"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


@st.cache_data(show_spinner=False)
def load_hospital_stats() -> pd.DataFrame:
    """Load per-hospital statistics from the raw CSV data files."""
    stats = []
    for i in range(1, 7):
        csv_path = DATA_DIR / f"hospital_{i}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            n_samples = len(df)
            n_positive = int(df["target"].sum())
            prevalence = n_positive / n_samples if n_samples > 0 else 0.0
            avg_age = float(df["age"].mean()) if "age" in df.columns else 0.0
            avg_chol = float(df["cholesterol"].mean()) if "cholesterol" in df.columns else 0.0
            avg_bp = float(df["resting_bp"].mean()) if "resting_bp" in df.columns else 0.0
            avg_hr = float(df["max_heart_rate"].mean()) if "max_heart_rate" in df.columns else 0.0
            avg_bmi = float(df["bmi"].mean()) if "bmi" in df.columns else 0.0
            smoker_pct = float(df["smoker"].mean()) * 100 if "smoker" in df.columns else 0.0
            stats.append({
                "Hospital": f"Hospital {i}",
                "ID": i,
                "Samples": n_samples,
                "Positive": n_positive,
                "Negative": n_samples - n_positive,
                "Prevalence": prevalence,
                "Avg_Age": avg_age,
                "Avg_Cholesterol": avg_chol,
                "Avg_BP": avg_bp,
                "Avg_HR": avg_hr,
                "Avg_BMI": avg_bmi,
                "Smoker_Pct": smoker_pct,
            })
    return pd.DataFrame(stats)


@st.cache_data(show_spinner=False)
def load_hospital_raw(hospital_id: int) -> Optional[pd.DataFrame]:
    """Load raw data for a specific hospital."""
    csv_path = DATA_DIR / f"hospital_{hospital_id}.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


@st.cache_data(show_spinner=False)
def load_combined_data() -> Optional[pd.DataFrame]:
    """Load the combined dataset."""
    path = DATA_DIR / "combined.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data(show_spinner=False)
def load_fedavg_rounds() -> Optional[pd.DataFrame]:
    """Load FedAvg round-by-round convergence data."""
    path = RESULTS_DIR / "rounds_fedavg.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data(show_spinner=False)
def load_attack_defense_matrix() -> Optional[pd.DataFrame]:
    """Load Phase 4 attack-defense experiment results."""
    path = RESULTS_DIR / "phase4_attack_defense_matrix.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data(show_spinner=False)
def load_dp_sweep() -> Optional[pd.DataFrame]:
    """Load Differential Privacy sweep results."""
    path = RESULTS_DIR / "phase4_dp_sweep.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data(show_spinner=False)
def load_non_iid_results() -> Optional[pd.DataFrame]:
    """Load Non-IID experiment results."""
    path = RESULTS_DIR / "phase3_non_iid_experiments.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data(show_spinner=False)
def load_fedprox_results() -> Optional[pd.DataFrame]:
    """Load FedProx experiment results."""
    path = RESULTS_DIR / "phase3_fedprox_experiments.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data(show_spinner=False)
def load_comm_cost() -> Optional[pd.DataFrame]:
    """Load communication cost analysis results."""
    path = RESULTS_DIR / "phase4_comm_cost.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data(show_spinner=False)
def load_attack_trajectories() -> Optional[pd.DataFrame]:
    """Load round-by-round attack trajectory data."""
    path = RESULTS_DIR / "phase4_attack_trajectories.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_resource(show_spinner=False)
def load_global_model():
    """Load the trained global FedCare model for real-time inference."""
    import sys
    import torch
    sys.path.insert(0, str(PROJECT_ROOT))
    from fedcare.task import Net
    from sklearn.preprocessing import StandardScaler

    model = Net()
    model.eval()

    checkpoint_path = CHECKPOINT_DIR / "final_fedavg_model.pt"
    if checkpoint_path.exists():
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        from fedcare.task import load_data
        _, _, scaler = load_data(partition_id=None)
    else:
        from fedcare.task import load_data, train as train_fn
        train_loader, _, scaler = load_data(partition_id=None)
        train_fn(model, train_loader, epochs=15, lr=0.001)
        model.eval()

    return model, scaler

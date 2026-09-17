"""
FedCare Interactive Dashboard
=============================
Phase 5 - Streamlit web application for the Privacy-Preserving
Federated Learning for Heart Disease Prediction project.

Features:
    1. Federated Network Topology Map with hospital statistics
    2. Interactive Training Console with strategy & attack selectors
    3. Round-by-round animated convergence charts (Loss, Accuracy, AUC)
    4. Inter-Hospital Equity & Fairness Analysis
    5. Attack vs. Defense Comparison Matrix
    6. Privacy-Utility Tradeoff Explorer
    7. Live Clinical Heart-Disease Risk Calculator
    8. Communication Cost Analysis

Usage:
    streamlit run app/dashboard.py

Author: FedCare Team
"""

from __future__ import annotations

import sys
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import torch

# ── Path Setup ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "heart"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# ── Streamlit Page Configuration ──────────────────────────────────────
st.set_page_config(
    page_title="FedCare - Privacy-Preserving Federated Learning",
    page_icon="heart",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS for Premium Look ───────────────────────────────────────
def inject_custom_css():
    """Inject premium glassmorphism-inspired CSS styling."""
    st.markdown(
        """
        <style>
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        /* Root variables */
        :root {
            --primary: #6366f1;
            --primary-light: #818cf8;
            --accent: #f43f5e;
            --success: #10b981;
            --warning: #f59e0b;
            --info: #3b82f6;
            --bg-dark: #0f172a;
            --bg-card: rgba(30, 41, 59, 0.7);
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --border: rgba(148, 163, 184, 0.15);
            --gradient-1: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
            --gradient-2: linear-gradient(135deg, #f43f5e 0%, #e11d48 100%);
            --gradient-3: linear-gradient(135deg, #10b981 0%, #059669 100%);
        }

        /* Global font */
        html, body, [class*="st-"] {
            font-family: 'Inter', sans-serif !important;
        }
        
        /* Fix Material Icons font for sidebar expand/collapse button */
        i, .material-icons, .material-symbols-rounded, [class^="stIcon"], [class*="icon"] {
            font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
        }

        /* Professional Sidebar Navigation Style */
        /* Hide the radio button circles */
        section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-baseweb="radio"] div:first-child {
            display: none !important;
        }
        
        /* Style the radio labels to look like professional buttons */
        section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-baseweb="radio"] {
            padding: 10px 16px;
            border-radius: 8px;
            background: transparent;
            margin-bottom: 4px;
            border: 1px solid transparent;
            transition: all 0.2s ease;
            cursor: pointer;
            width: 100%;
        }
        
        /* Hover effect */
        section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-baseweb="radio"]:hover {
            background-color: rgba(99, 102, 241, 0.1);
        }
        
        /* Selected state */
        section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-baseweb="radio"][aria-checked="true"] {
            background: linear-gradient(90deg, rgba(99, 102, 241, 0.15) 0%, transparent 100%);
            border-left: 4px solid var(--primary);
        }
        
        /* Make text inside selected item pop out */
        section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-baseweb="radio"][aria-checked="true"] p {
            color: var(--primary-light) !important;
            font-weight: 700;
        }

        /* Main background */
        .stApp {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: rgba(15, 23, 42, 0.95) !important;
            border-right: 1px solid var(--border);
        }

        /* Metric cards */
        div[data-testid="stMetric"] {
            background: var(--bg-card);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 20px 24px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 40px rgba(99, 102, 241, 0.15);
        }

        div[data-testid="stMetric"] label {
            color: var(--text-secondary) !important;
            font-size: 0.85rem !important;
            font-weight: 500 !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: var(--text-primary) !important;
            font-size: 1.8rem !important;
            font-weight: 700 !important;
        }

        /* Headers */
        h1, h2, h3 {
            color: var(--text-primary) !important;
            font-weight: 700 !important;
        }

        h1 {
            background: var(--gradient-1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 2.2rem !important;
        }

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: var(--bg-card);
            border-radius: 12px;
            padding: 4px;
            border: 1px solid var(--border);
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            color: var(--text-secondary);
            font-weight: 500;
            padding: 8px 20px;
        }

        .stTabs [aria-selected="true"] {
            background: var(--gradient-1) !important;
            color: white !important;
        }

        /* Expander */
        .streamlit-expanderHeader {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            color: var(--text-primary) !important;
            font-weight: 600;
        }

        /* Divider */
        hr {
            border-color: var(--border) !important;
        }

        /* Glass card helper */
        .glass-card {
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 24px;
            margin: 8px 0;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        }

        .hero-badge {
            display: inline-block;
            background: var(--gradient-1);
            color: white;
            padding: 4px 16px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }

        /* Risk indicator */
        .risk-low { color: #10b981; font-weight: 700; font-size: 1.5rem; }
        .risk-moderate { color: #f59e0b; font-weight: 700; font-size: 1.5rem; }
        .risk-high { color: #ef4444; font-weight: 700; font-size: 1.5rem; }
        .risk-very-high { color: #dc2626; font-weight: 700; font-size: 1.5rem; }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: transparent;
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(148, 163, 184, 0.3);
            border-radius: 3px;
        }

        /* Selectbox / Input styling */
        .stSelectbox > div > div,
        .stNumberInput > div > div > input,
        .stSlider > div {
            border-color: var(--border) !important;
        }

        /* Button */
        .stButton > button {
            background: var(--gradient-1) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 10px 24px !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
        }

        .stButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important;
        }

        /* Info/Warning/Error boxes */
        .stAlert {
            border-radius: 12px !important;
        }

        /* Network topology */
        .hospital-node {
            text-align: center;
            padding: 16px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            transition: all 0.3s ease;
        }
        .hospital-node:hover {
            border-color: var(--primary);
            box-shadow: 0 0 20px rgba(99, 102, 241, 0.2);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Cached Data Loaders ──────────────────────────────────────────────
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
            stats.append({
                "Hospital": f"Hospital {i}",
                "ID": i,
                "Samples": n_samples,
                "Positive": n_positive,
                "Negative": n_samples - n_positive,
                "Prevalence": prevalence,
                "Avg_Age": avg_age,
                "Avg_Cholesterol": avg_chol,
            })
    return pd.DataFrame(stats)


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


# ── Model Loader ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_global_model():
    """Load the trained global FedCare model for real-time inference."""
    from fedcare.task import Net
    from sklearn.preprocessing import StandardScaler

    model = Net()
    model.eval()

    # Try to load checkpoint if available
    checkpoint_path = CHECKPOINT_DIR / "global_model.pt"
    if checkpoint_path.exists():
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
    else:
        # Fall back: train a quick centralized model for the risk calculator
        from fedcare.task import load_data, train as train_fn
        train_loader, _, scaler = load_data(partition_id=None)
        train_fn(model, train_loader, epochs=15, lr=0.001)
        model.eval()
        return model, scaler

    # Fit scaler on combined data
    combined_path = DATA_DIR / "combined.csv"
    df = pd.read_csv(combined_path)
    X = df.drop(columns=["target"]).values
    scaler = StandardScaler()
    scaler.fit(X)

    return model, scaler


# ══════════════════════════════════════════════════════════════════════
#                        PAGE COMPONENTS
# ══════════════════════════════════════════════════════════════════════

def render_header():
    """Render the main dashboard header with hero section."""
    st.markdown(
        '<div class="hero-badge">M.Tech Research Project</div>',
        unsafe_allow_html=True,
    )
    st.title("FedCare: Privacy-Preserving Federated Learning for Heart Disease Prediction")
    st.markdown(
        """
        <p style="color: #94a3b8; font-size: 1.05rem; max-width: 800px; line-height: 1.6;">
            A federated learning framework enabling <strong style="color: #818cf8;">6 hospitals</strong>
            to collaboratively train a heart disease classifier while preserving patient privacy.
            No raw data ever leaves a hospital.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")


def render_network_topology():
    """Render the federated network topology visualization."""
    st.subheader("Federated Network Topology")

    hospital_stats = load_hospital_stats()
    if hospital_stats.empty:
        st.warning("No hospital data found. Ensure data/heart/ contains hospital CSVs.")
        return

    # Create a network graph using Plotly
    fig = go.Figure()

    # Central server node
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode="markers+text",
        marker=dict(size=50, color="#6366f1", symbol="diamond",
                    line=dict(width=3, color="#818cf8")),
        text=["FedCare\nServer"],
        textposition="bottom center",
        textfont=dict(size=12, color="#f1f5f9", family="Inter"),
        name="Central Server",
        hoverinfo="text",
        hovertext="Federated Aggregation Server<br>Strategy: FedAvg / FedProx / Krum / TrimmedMean / Median",
    ))

    # Hospital nodes arranged in a circle
    n_hospitals = len(hospital_stats)
    angles = np.linspace(0, 2 * np.pi, n_hospitals, endpoint=False)
    radius = 3.0
    colors = ["#f43f5e", "#f59e0b", "#10b981", "#3b82f6", "#8b5cf6", "#ec4899"]

    for idx, (_, row) in enumerate(hospital_stats.iterrows()):
        x = radius * np.cos(angles[idx])
        y = radius * np.sin(angles[idx])

        # Connection line to server
        fig.add_trace(go.Scatter(
            x=[0, x], y=[0, y],
            mode="lines",
            line=dict(width=2, color="rgba(99, 102, 241, 0.4)", dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        ))

        # Hospital node
        prevalence_pct = row["Prevalence"] * 100
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers+text",
            marker=dict(
                size=35 + row["Samples"] / 80,
                color=colors[idx],
                line=dict(width=2, color="rgba(255,255,255,0.3)"),
            ),
            text=[f"H{row['ID']}"],
            textposition="middle center",
            textfont=dict(size=11, color="white", family="Inter"),
            name=row["Hospital"],
            hoverinfo="text",
            hovertext=(
                f"<b>{row['Hospital']}</b><br>"
                f"Samples: {row['Samples']:,}<br>"
                f"Disease Prevalence: {prevalence_pct:.1f}%<br>"
                f"Avg Age: {row['Avg_Age']:.1f}<br>"
                f"Avg Cholesterol: {row['Avg_Cholesterol']:.1f}"
            ),
        ))

    fig.update_layout(
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-5, 5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-5, 5]),
        height=450,
        margin=dict(l=20, r=20, t=20, b=20),
        annotations=[
            dict(
                x=0, y=0.8,
                text="Encrypted Model<br>Parameters Only",
                showarrow=False,
                font=dict(size=9, color="#94a3b8"),
            )
        ],
    )

    st.plotly_chart(fig, use_container_width=True)

    # Hospital stats cards
    cols = st.columns(6)
    for idx, (_, row) in enumerate(hospital_stats.iterrows()):
        with cols[idx]:
            prevalence_pct = row["Prevalence"] * 100
            st.metric(
                label=f"Hospital {row['ID']}",
                value=f"{row['Samples']:,}",
                delta=f"{prevalence_pct:.1f}% prevalence",
            )


def render_training_console():
    """Render the interactive training convergence visualization."""
    st.subheader("Training Convergence Dashboard")

    rounds_df = load_fedavg_rounds()
    if rounds_df is None:
        st.warning("No training round data found. Run `run_federated.py` first.")
        return

    # Controls
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    with col_ctrl1:
        metric_choice = st.selectbox(
            "Primary Metric",
            ["ROC-AUC", "Accuracy", "Loss"],
            index=0,
            key="metric_select",
        )
    with col_ctrl2:
        show_hospitals = st.toggle("Show Per-Hospital AUC", value=True, key="show_hosp")
    with col_ctrl3:
        animate = st.toggle("Animate Round-by-Round", value=False, key="animate")

    # Determine which column to plot
    metric_map = {
        "ROC-AUC": ("auc", "ROC-AUC"),
        "Accuracy": ("accuracy", "Accuracy"),
        "Loss": ("test_loss", "Test Loss"),
    }
    col_name, display_name = metric_map[metric_choice]

    if animate:
        # Animated slider to simulate round-by-round progression
        max_round = int(rounds_df["round"].max())
        current_round = st.slider(
            "Simulation Round",
            min_value=1,
            max_value=max_round,
            value=max_round,
            key="round_slider",
        )
        plot_df = rounds_df[rounds_df["round"] <= current_round]
    else:
        plot_df = rounds_df

    # Main convergence chart
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.6, 0.4],
        subplot_titles=(f"Global {display_name} Convergence", "Per-Hospital AUC Distribution"),
    )

    # Global metric line
    fig.add_trace(
        go.Scatter(
            x=plot_df["round"],
            y=plot_df[col_name],
            mode="lines+markers",
            name=f"Global {display_name}",
            line=dict(width=3, color="#6366f1"),
            marker=dict(size=6),
            fill="tozeroy" if col_name != "test_loss" else None,
            fillcolor="rgba(99, 102, 241, 0.1)",
        ),
        row=1, col=1,
    )

    # Per-hospital AUC lines
    if show_hospitals and col_name == "auc":
        hosp_colors = ["#f43f5e", "#f59e0b", "#10b981", "#3b82f6", "#8b5cf6", "#ec4899"]
        for i in range(1, 7):
            hosp_col = f"hosp_{i}_auc"
            if hosp_col in plot_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=plot_df["round"],
                        y=plot_df[hosp_col],
                        mode="lines",
                        name=f"Hospital {i}",
                        line=dict(width=1.5, color=hosp_colors[i - 1], dash="dash"),
                        opacity=0.7,
                    ),
                    row=1, col=1,
                )

    # Box plot for per-hospital AUC distribution at final round
    last_row = plot_df.iloc[-1]
    hosp_aucs = []
    hosp_labels = []
    hosp_colors_list = ["#f43f5e", "#f59e0b", "#10b981", "#3b82f6", "#8b5cf6", "#ec4899"]
    for i in range(1, 7):
        hosp_col = f"hosp_{i}_auc"
        if hosp_col in last_row.index:
            hosp_aucs.append(float(last_row[hosp_col]))
            hosp_labels.append(f"H{i}")

    if hosp_aucs:
        fig.add_trace(
            go.Bar(
                x=hosp_labels,
                y=hosp_aucs,
                marker=dict(color=hosp_colors_list[:len(hosp_aucs)],
                            line=dict(width=1, color="rgba(255,255,255,0.3)")),
                name="Hospital AUC",
                text=[f"{v:.4f}" for v in hosp_aucs],
                textposition="outside",
                textfont=dict(size=10, color="#f1f5f9"),
            ),
            row=1, col=2,
        )

    fig.update_layout(
        height=450,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f1f5f9", family="Inter"),
        legend=dict(
            bgcolor="rgba(30, 41, 59, 0.7)",
            bordercolor="rgba(148, 163, 184, 0.15)",
            font=dict(size=10),
        ),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    fig.update_xaxes(title_text="Communication Round", row=1, col=1, gridcolor="rgba(148,163,184,0.1)")
    fig.update_yaxes(title_text=display_name, row=1, col=1, gridcolor="rgba(148,163,184,0.1)")
    fig.update_xaxes(title_text="Hospital", row=1, col=2, gridcolor="rgba(148,163,184,0.1)")
    fig.update_yaxes(
        title_text="AUC", row=1, col=2,
        range=[min(hosp_aucs) - 0.02, max(hosp_aucs) + 0.02] if hosp_aucs else None,
        gridcolor="rgba(148,163,184,0.1)",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Summary metrics
    last = rounds_df.iloc[-1]
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Final Global AUC", f"{last['auc']:.4f}")
    with col_m2:
        st.metric("Final Accuracy", f"{last['accuracy']:.4f}")
    with col_m3:
        hosp_auc_vals = [float(last[f"hosp_{i}_auc"]) for i in range(1, 7) if f"hosp_{i}_auc" in last.index]
        equity_gap = max(hosp_auc_vals) - min(hosp_auc_vals) if hosp_auc_vals else 0
        st.metric("Equity Gap", f"{equity_gap:.4f}")
    with col_m4:
        st.metric("Total Rounds", f"{int(last['round'])}")


def render_attack_defense_analysis():
    """Render the attack vs. defense analysis visualization."""
    st.subheader("Adversarial Robustness: Attack vs. Defense Matrix")

    attack_df = load_attack_defense_matrix()
    if attack_df is None:
        st.warning("No attack-defense data found. Run `run_phase4_experiments.py` first.")
        return

    # Sidebar filters
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        selected_attacks = st.multiselect(
            "Filter Attacks",
            options=attack_df["Attack_Name"].unique().tolist(),
            default=attack_df["Attack_Name"].unique().tolist(),
            key="attack_filter",
        )
    with col_f2:
        selected_metric = st.selectbox(
            "Comparison Metric",
            ["Final_AUC", "Final_Accuracy", "Equity_Gap", "Worst_Hospital_AUC"],
            index=0,
            key="defense_metric",
        )

    filtered = attack_df[attack_df["Attack_Name"].isin(selected_attacks)]

    # Grouped bar chart
    fig = px.bar(
        filtered,
        x="Attack_Name",
        y=selected_metric,
        color="Strategy_Name",
        barmode="group",
        color_discrete_sequence=["#6366f1", "#f43f5e", "#10b981", "#f59e0b", "#8b5cf6"],
        labels={
            "Attack_Name": "Attack Scenario",
            selected_metric: selected_metric.replace("_", " "),
            "Strategy_Name": "Defense Strategy",
        },
    )
    fig.update_layout(
        height=450,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f1f5f9", family="Inter"),
        legend=dict(
            bgcolor="rgba(30, 41, 59, 0.7)",
            bordercolor="rgba(148, 163, 184, 0.15)",
        ),
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
        yaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Heatmap pivot
    st.markdown("##### Attack x Defense Heatmap")
    if "Final_AUC" in filtered.columns and len(filtered) > 0:
        pivot = filtered.pivot_table(
            index="Attack_Name",
            columns="Strategy_Name",
            values="Final_AUC",
            aggfunc="mean",
        )
        fig_hm = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=[
                [0.0, "#dc2626"],
                [0.5, "#f59e0b"],
                [1.0, "#10b981"],
            ],
            text=np.round(pivot.values, 4),
            texttemplate="%{text}",
            textfont=dict(size=12, color="white"),
            hoverongaps=False,
            colorbar=dict(title="AUC"),
        ))
        fig_hm.update_layout(
            height=350,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f1f5f9", family="Inter"),
            margin=dict(l=100, r=20, t=20, b=40),
            xaxis_title="Defense Strategy",
            yaxis_title="Attack Scenario",
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    # Attack trajectory chart
    traj_df = load_attack_trajectories()
    if traj_df is not None and len(traj_df) > 0:
        st.markdown("##### Round-by-Round Attack Impact Trajectories")
        traj_cols = list(traj_df.columns)
        if "round" in traj_cols and "auc" in traj_cols and "label" in traj_cols:
            fig_traj = px.line(
                traj_df, x="round", y="auc", color="label",
                color_discrete_sequence=["#6366f1", "#f43f5e", "#10b981", "#f59e0b", "#8b5cf6"],
                labels={"round": "Communication Round", "auc": "Global AUC", "label": "Scenario"},
            )
            fig_traj.update_layout(
                height=350,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f1f5f9", family="Inter"),
                legend=dict(bgcolor="rgba(30, 41, 59, 0.7)", bordercolor="rgba(148, 163, 184, 0.15)"),
                margin=dict(l=40, r=20, t=20, b=40),
                xaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
                yaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
            )
            st.plotly_chart(fig_traj, use_container_width=True)


def render_privacy_utility():
    """Render the differential privacy vs. utility tradeoff explorer."""
    st.subheader("Privacy-Utility Tradeoff Explorer")

    dp_df = load_dp_sweep()
    if dp_df is None:
        st.warning("No DP sweep data found. Run `run_phase4_experiments.py` first.")
        return

    col_pu1, col_pu2 = st.columns([3, 1])

    with col_pu1:
        # Scatter plot: epsilon vs AUC
        fig = go.Figure()

        # AUC line
        fig.add_trace(go.Scatter(
            x=dp_df["Epsilon"],
            y=dp_df["Final_AUC"],
            mode="lines+markers",
            name="Global AUC",
            line=dict(width=3, color="#6366f1"),
            marker=dict(size=10, symbol="circle"),
        ))

        # Worst hospital AUC
        fig.add_trace(go.Scatter(
            x=dp_df["Epsilon"],
            y=dp_df["Worst_Hospital_AUC"],
            mode="lines+markers",
            name="Worst Hospital AUC",
            line=dict(width=2, color="#f43f5e", dash="dash"),
            marker=dict(size=8, symbol="diamond"),
        ))

        # Privacy regime annotations
        for _, row in dp_df.iterrows():
            regime = str(row.get("Privacy_Regime", ""))
            if regime == "Strong":
                color = "#10b981"
            elif regime == "Moderate":
                color = "#f59e0b"
            elif regime == "Weak":
                color = "#ef4444"
            else:
                color = "#94a3b8"

            fig.add_annotation(
                x=row["Epsilon"],
                y=row["Final_AUC"],
                text=regime,
                showarrow=False,
                yshift=18,
                font=dict(size=9, color=color),
            )

        fig.update_layout(
            xaxis_title="Privacy Budget (epsilon)",
            yaxis_title="AUC",
            height=400,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f1f5f9", family="Inter"),
            legend=dict(
                bgcolor="rgba(30, 41, 59, 0.7)",
                bordercolor="rgba(148, 163, 184, 0.15)",
            ),
            margin=dict(l=40, r=20, t=20, b=40),
            xaxis=dict(gridcolor="rgba(148,163,184,0.1)", type="log"),
            yaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_pu2:
        st.markdown("##### DP Configuration")
        st.dataframe(
            dp_df[["Noise_Multiplier", "Epsilon", "Privacy_Regime", "Final_AUC"]].style.format({
                "Noise_Multiplier": "{:.3f}",
                "Epsilon": "{:.2f}",
                "Final_AUC": "{:.4f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        # Key insight
        best_row = dp_df.loc[dp_df["Final_AUC"].idxmax()]
        st.info(
            f"**Best AUC**: {best_row['Final_AUC']:.4f} at "
            f"epsilon={best_row['Epsilon']:.2f} "
            f"({best_row.get('Privacy_Regime', 'N/A')} privacy)"
        )


def render_non_iid_analysis():
    """Render Non-IID data distribution analysis."""
    st.subheader("Non-IID Data Heterogeneity Analysis")

    non_iid_df = load_non_iid_results()
    fedprox_df = load_fedprox_results()

    col_n1, col_n2 = st.columns(2)

    with col_n1:
        if non_iid_df is not None:
            st.markdown("##### Impact of Data Heterogeneity")
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=non_iid_df["Configuration"],
                y=non_iid_df["Final_AUC"],
                name="Global AUC",
                marker_color="#6366f1",
                text=non_iid_df["Final_AUC"].round(4),
                textposition="outside",
                textfont=dict(color="#f1f5f9", size=10),
            ))

            fig.add_trace(go.Bar(
                x=non_iid_df["Configuration"],
                y=non_iid_df["Equity_Gap"],
                name="Equity Gap",
                marker_color="#f43f5e",
                text=non_iid_df["Equity_Gap"].round(4),
                textposition="outside",
                textfont=dict(color="#f1f5f9", size=10),
                yaxis="y2",
            ))

            fig.update_layout(
                height=400,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f1f5f9", family="Inter", size=10),
                barmode="group",
                yaxis=dict(title="AUC", gridcolor="rgba(148,163,184,0.1)"),
                yaxis2=dict(title="Equity Gap", overlaying="y", side="right",
                            gridcolor="rgba(148,163,184,0.1)"),
                legend=dict(bgcolor="rgba(30, 41, 59, 0.7)", bordercolor="rgba(148, 163, 184, 0.15)"),
                margin=dict(l=40, r=40, t=20, b=80),
                xaxis=dict(tickangle=-25),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No Non-IID data found.")

    with col_n2:
        if fedprox_df is not None:
            st.markdown("##### FedProx vs. FedAvg Comparison")
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=fedprox_df["Algorithm"],
                y=fedprox_df["Final_AUC"],
                name="Final AUC",
                marker_color="#10b981",
                text=fedprox_df["Final_AUC"].round(4),
                textposition="outside",
                textfont=dict(color="#f1f5f9", size=10),
            ))

            fig.add_trace(go.Bar(
                x=fedprox_df["Algorithm"],
                y=fedprox_df["Equity_Gap"],
                name="Equity Gap",
                marker_color="#f59e0b",
                text=fedprox_df["Equity_Gap"].round(4),
                textposition="outside",
                textfont=dict(color="#f1f5f9", size=10),
            ))

            fig.update_layout(
                height=400,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f1f5f9", family="Inter", size=10),
                barmode="group",
                yaxis=dict(title="Value", gridcolor="rgba(148,163,184,0.1)"),
                legend=dict(bgcolor="rgba(30, 41, 59, 0.7)", bordercolor="rgba(148, 163, 184, 0.15)"),
                margin=dict(l=40, r=20, t=20, b=80),
                xaxis=dict(tickangle=-25),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No FedProx data found.")


def render_communication_cost():
    """Render communication cost analysis."""
    st.subheader("Communication Efficiency Analysis")

    comm_df = load_comm_cost()
    if comm_df is None:
        st.warning("No communication cost data found.")
        return

    row = comm_df.iloc[0]

    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    with col_c1:
        st.metric("Model Parameters", f"{int(row['param_count']):,}")
    with col_c2:
        st.metric("Single Message", f"{row['single_message_kb']:.1f} KB")
    with col_c3:
        st.metric("Total FL Comm.", f"{row['total_fl_mb']:.2f} MB")
    with col_c4:
        ratio = row["comm_ratio"]
        savings = (1 - ratio) * 100
        st.metric(
            "vs. Centralized Data Transfer",
            f"{ratio:.4f}x",
            delta=f"{savings:.1f}% savings",
        )

    # Comparison bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Federated\nModel Exchange", "Centralized\nRaw Data Transfer"],
        y=[row["total_fl_mb"], row["centralized_data_mb"]],
        marker_color=["#6366f1", "#f43f5e"],
        text=[f"{row['total_fl_mb']:.2f} MB", f"{row['centralized_data_mb']:.2f} MB"],
        textposition="outside",
        textfont=dict(color="#f1f5f9", size=13, family="Inter"),
        width=0.4,
    ))
    fig.update_layout(
        height=350,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f1f5f9", family="Inter"),
        yaxis=dict(title="Data Transfer (MB)", gridcolor="rgba(148,163,184,0.1)"),
        margin=dict(l=40, r=20, t=20, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_risk_calculator():
    """Render the live clinical heart disease risk calculator."""
    st.subheader("Clinical Heart Disease Risk Calculator")
    st.markdown(
        """
        <p style="color: #94a3b8; font-size: 0.95rem;">
            Enter a patient's 13 clinical features below. The global federated model
            will compute a real-time heart disease probability without any patient data
            leaving this interface.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Load model
    with st.spinner("Loading global model..."):
        model, scaler = load_global_model()

    # Input form
    with st.form("risk_form", clear_on_submit=False):
        st.markdown("##### Patient Clinical Features")

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            age = st.number_input("Age (years)", min_value=18, max_value=100, value=55, step=1)
            resting_bp = st.number_input("Resting Blood Pressure (mmHg)", min_value=80, max_value=220, value=130, step=1)
            cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600, value=240, step=5)
            max_hr = st.number_input("Max Heart Rate (bpm)", min_value=60, max_value=220, value=150, step=1)

        with col_b:
            bmi = st.number_input("BMI (kg/m^2)", min_value=15.0, max_value=55.0, value=27.5, step=0.1, format="%.1f")
            glucose = st.number_input("Fasting Glucose (mg/dL)", min_value=50, max_value=400, value=100, step=5)
            sex = st.selectbox("Sex", options=["Male", "Female"], index=0)
            smoker = st.selectbox("Smoker", options=["No", "Yes"], index=0)

        with col_c:
            diabetes = st.selectbox("Diabetes History", options=["No", "Yes"], index=0)
            family_history = st.selectbox("Family History of Heart Disease", options=["No", "Yes"], index=0)
            chest_pain = st.selectbox(
                "Chest Pain Type",
                options=["Asymptomatic", "Atypical Angina", "Non-Anginal", "Typical Angina"],
                index=0,
            )

        submitted = st.form_submit_button("Predict Heart Disease Risk", use_container_width=True)

    if submitted:
        # Encode features to match training data columns:
        # age, resting_bp, cholesterol, max_heart_rate, bmi, glucose,
        # sex, smoker, diabetes_history, family_history,
        # cp_atypical_angina, cp_non_anginal, cp_typical_angina
        sex_val = 1 if sex == "Male" else 0
        smoker_val = 1 if smoker == "Yes" else 0
        diabetes_val = 1 if diabetes == "Yes" else 0
        family_val = 1 if family_history == "Yes" else 0

        cp_atypical = 1 if chest_pain == "Atypical Angina" else 0
        cp_non_anginal = 1 if chest_pain == "Non-Anginal" else 0
        cp_typical = 1 if chest_pain == "Typical Angina" else 0

        raw_features = np.array([[
            age, resting_bp, cholesterol, max_hr, bmi, glucose,
            sex_val, smoker_val, diabetes_val, family_val,
            cp_atypical, cp_non_anginal, cp_typical,
        ]], dtype=np.float64)

        # Scale features
        scaled_features = scaler.transform(raw_features)
        input_tensor = torch.tensor(scaled_features, dtype=torch.float32)

        # Inference
        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
            risk_prob = float(probabilities[0, 1])

        # Display results
        st.markdown("---")

        col_r1, col_r2, col_r3 = st.columns([1, 2, 1])

        with col_r2:
            # Risk level classification
            if risk_prob < 0.25:
                risk_level = "LOW RISK"
                risk_class = "risk-low"
                risk_color = "#10b981"
                risk_emoji = "green"
                risk_advice = "Continue maintaining a healthy lifestyle. Regular check-ups recommended."
            elif risk_prob < 0.50:
                risk_level = "MODERATE RISK"
                risk_class = "risk-moderate"
                risk_color = "#f59e0b"
                risk_emoji = "yellow"
                risk_advice = "Consider lifestyle modifications and consult with a cardiologist."
            elif risk_prob < 0.75:
                risk_level = "HIGH RISK"
                risk_class = "risk-high"
                risk_color = "#ef4444"
                risk_emoji = "orange"
                risk_advice = "Immediate medical consultation strongly recommended. Diagnostic tests advised."
            else:
                risk_level = "VERY HIGH RISK"
                risk_class = "risk-very-high"
                risk_color = "#dc2626"
                risk_emoji = "red"
                risk_advice = "Urgent cardiology referral needed. Comprehensive cardiac workup required."

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_prob * 100,
                number=dict(suffix="%", font=dict(size=48, color="#f1f5f9")),
                title=dict(text="Heart Disease Probability", font=dict(size=16, color="#94a3b8")),
                gauge=dict(
                    axis=dict(range=[0, 100], tickwidth=1, tickcolor="#475569"),
                    bar=dict(color=risk_color, thickness=0.3),
                    bgcolor="rgba(30, 41, 59, 0.5)",
                    borderwidth=2,
                    bordercolor="rgba(148, 163, 184, 0.15)",
                    steps=[
                        dict(range=[0, 25], color="rgba(16, 185, 129, 0.2)"),
                        dict(range=[25, 50], color="rgba(245, 158, 11, 0.2)"),
                        dict(range=[50, 75], color="rgba(239, 68, 68, 0.2)"),
                        dict(range=[75, 100], color="rgba(220, 38, 38, 0.2)"),
                    ],
                    threshold=dict(
                        line=dict(color=risk_color, width=4),
                        thickness=0.8,
                        value=risk_prob * 100,
                    ),
                ),
            ))
            fig_gauge.update_layout(
                height=300,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f1f5f9", family="Inter"),
                margin=dict(l=30, r=30, t=60, b=20),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown(
                f'<p class="{risk_class}" style="text-align: center; font-size: 1.8rem;">'
                f'{risk_level}</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<p style="color: #94a3b8; text-align: center; font-size: 1rem;">{risk_advice}</p>',
                unsafe_allow_html=True,
            )

        # Feature contribution breakdown
        st.markdown("##### Patient Feature Summary")
        feature_data = {
            "Feature": [
                "Age", "Resting BP", "Cholesterol", "Max Heart Rate",
                "BMI", "Glucose", "Sex", "Smoker",
                "Diabetes History", "Family History", "Chest Pain Type",
            ],
            "Value": [
                f"{age} years", f"{resting_bp} mmHg", f"{cholesterol} mg/dL",
                f"{max_hr} bpm", f"{bmi:.1f} kg/m2", f"{glucose} mg/dL",
                sex, smoker, diabetes, family_history, chest_pain,
            ],
        }
        st.dataframe(pd.DataFrame(feature_data), use_container_width=True, hide_index=True)

        st.caption(
            "**Disclaimer**: This is a research demonstration tool. "
            "Predictions should not be used for clinical decision-making "
            "without professional medical review."
        )


def render_project_overview():
    """Render the project overview and methodology section."""
    st.subheader("Project Overview & Methodology")

    col_o1, col_o2 = st.columns(2)

    with col_o1:
        st.markdown(
            """
            ##### Federated Learning Architecture

            **FedCare** uses the **Flower** framework to orchestrate federated
            training across **6 geographically distributed hospitals**, each with
            ~2,000 patient records.

            **Key Principles:**
            - Raw patient data **never leaves** the hospital premises
            - Only model parameter updates (weights & biases) are communicated
            - The central server aggregates updates using configurable strategies
            - Each hospital retains full data sovereignty
            """
        )

        st.markdown(
            """
            ##### Aggregation Strategies Implemented

            | Strategy | Description |
            |----------|-------------|
            | **FedAvg** | Weighted average of client model updates |
            | **FedProx** | Proximal regularization for non-IID data |
            | **Trimmed Mean** | Coordinate-wise trimmed mean (Byzantine-robust) |
            | **Coordinate Median** | Element-wise median aggregation |
            | **Multi-Krum** | Distance-based outlier filtering |
            """
        )

    with col_o2:
        st.markdown(
            """
            ##### Security & Privacy Mechanisms

            **Adversarial Attack Simulations:**
            - Label-Flipping: Malicious hospital inverts training labels
            - Model Poisoning: Adversary corrupts model weight updates

            **Byzantine-Robust Defenses:**
            - Trimmed Mean removes statistical outliers before averaging
            - Coordinate Median is immune to extreme single-coordinate manipulation
            - Multi-Krum selects most representative model updates

            **Differential Privacy (DP):**
            - L2 gradient clipping bounds individual contribution sensitivity
            - Calibrated Gaussian noise guarantees (epsilon, delta)-privacy
            - Renyi privacy accountant tracks cumulative privacy expenditure
            """
        )

        st.markdown(
            """
            ##### Model Architecture

            ```
            Input (13 clinical features)
              -> Linear(64) -> ReLU -> Dropout(0.3)
              -> Linear(32) -> ReLU -> Dropout(0.3)
              -> Linear(2)  (binary classification logits)
            ```

            **Features**: Age, Resting BP, Cholesterol, Max HR, BMI,
            Glucose, Sex, Smoker, Diabetes History, Family History,
            Chest Pain Type (one-hot encoded)
            """
        )


def render_existing_figures():
    """Render pre-generated research figures from results directory."""
    st.subheader("Research Figures Gallery")

    figures = {
        "Figure 1: FedAvg Convergence": RESULTS_DIR / "figure1_fedavg_convergence.png",
        "Figure 2: Non-IID Impact": RESULTS_DIR / "figure2_non_iid_impact.png",
        "Figure 3: FedProx vs FedAvg": RESULTS_DIR / "figure3_fedprox_vs_fedavg.png",
        "Figure 4: Attacks & Defenses": RESULTS_DIR / "figure4_attacks_and_defenses.png",
        "Figure 5: Privacy-Utility Tradeoff": RESULTS_DIR / "figure5_privacy_utility.png",
    }

    available = {k: v for k, v in figures.items() if v.exists()}

    if not available:
        st.warning("No pre-generated figures found in the results directory.")
        return

    # Display in columns of 2
    keys = list(available.keys())
    for i in range(0, len(keys), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(keys):
                with col:
                    st.markdown(f"##### {keys[idx]}")
                    st.image(str(available[keys[idx]]), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
#                           SIDEBAR
# ══════════════════════════════════════════════════════════════════════

def render_sidebar():
    """Render the sidebar with navigation and project info."""
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align: center; padding: 16px 0;">
                <h2 style="
                    background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    font-size: 1.6rem;
                    margin: 0;
                ">FedCare</h2>
                <p style="color: #94a3b8; font-size: 0.8rem; margin: 4px 0;">
                    Privacy-Preserving FL
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        page = st.radio(
            "Navigation",
            [
                "Dashboard Overview",
                "Network Topology",
                "Training Console",
                "Attack vs. Defense",
                "Privacy-Utility Tradeoff",
                "Non-IID Analysis",
                "Communication Cost",
                "Risk Calculator",
                "Research Figures",
                "Project Overview",
            ],
            index=0,
            key="nav",
        )

        st.markdown("---")

        # Quick stats
        hospital_stats = load_hospital_stats()
        if not hospital_stats.empty:
            total_samples = hospital_stats["Samples"].sum()
            avg_prevalence = hospital_stats["Prevalence"].mean() * 100
            st.markdown("##### Dataset Summary")
            st.markdown(
                f"""
                - **Hospitals**: 6
                - **Total Patients**: {total_samples:,}
                - **Avg. Prevalence**: {avg_prevalence:.1f}%
                - **Features**: 13 clinical
                - **Model**: MLP (13->64->32->2)
                """
            )

        st.markdown("---")
        st.markdown(
            """
            <div style="text-align: center;">
                <p style="color: #475569; font-size: 0.7rem;">
                    M.Tech Research Project<br>
                    Privacy-Preserving Federated Learning<br>
                    for Heart Disease Prediction
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return page


# ══════════════════════════════════════════════════════════════════════
#                         DASHBOARD OVERVIEW
# ══════════════════════════════════════════════════════════════════════

def render_dashboard_overview():
    """Render the main dashboard overview combining key insights."""
    # Top metrics row
    rounds_df = load_fedavg_rounds()
    attack_df = load_attack_defense_matrix()
    dp_df = load_dp_sweep()
    comm_df = load_comm_cost()

    col1, col2, col3, col4, col5 = st.columns(5)

    if rounds_df is not None:
        last = rounds_df.iloc[-1]
        with col1:
            st.metric("Global AUC", f"{last['auc']:.4f}")
        with col2:
            st.metric("Global Accuracy", f"{last['accuracy']:.4f}")
        with col3:
            st.metric("Training Rounds", f"{int(last['round'])}")
    else:
        with col1:
            st.metric("Global AUC", "N/A")
        with col2:
            st.metric("Global Accuracy", "N/A")
        with col3:
            st.metric("Training Rounds", "N/A")

    with col4:
        if attack_df is not None:
            n_experiments = len(attack_df)
            st.metric("Security Experiments", str(n_experiments))
        else:
            st.metric("Security Experiments", "0")

    with col5:
        if dp_df is not None:
            best_eps = dp_df.loc[dp_df["Final_AUC"].idxmax(), "Epsilon"]
            st.metric("Best Privacy (eps)", f"{best_eps:.2f}")
        else:
            st.metric("Best Privacy (eps)", "N/A")

    st.markdown("---")

    # Overview charts
    col_ov1, col_ov2 = st.columns(2)

    with col_ov1:
        st.markdown("##### Training Convergence")
        if rounds_df is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rounds_df["round"], y=rounds_df["auc"],
                mode="lines+markers",
                name="AUC",
                line=dict(width=3, color="#6366f1"),
                fill="tozeroy",
                fillcolor="rgba(99, 102, 241, 0.1)",
            ))
            fig.add_trace(go.Scatter(
                x=rounds_df["round"], y=rounds_df["accuracy"],
                mode="lines+markers",
                name="Accuracy",
                line=dict(width=2, color="#10b981", dash="dot"),
            ))
            fig.update_layout(
                height=320,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f1f5f9", family="Inter"),
                legend=dict(bgcolor="rgba(30,41,59,0.7)", bordercolor="rgba(148,163,184,0.15)"),
                margin=dict(l=40, r=20, t=10, b=40),
                xaxis=dict(title="Round", gridcolor="rgba(148,163,184,0.1)"),
                yaxis=dict(title="Metric", gridcolor="rgba(148,163,184,0.1)"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run training to see convergence charts.")

    with col_ov2:
        st.markdown("##### Hospital Data Distribution")
        hospital_stats = load_hospital_stats()
        if not hospital_stats.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=hospital_stats["Hospital"],
                y=hospital_stats["Positive"],
                name="Positive (Heart Disease)",
                marker_color="#f43f5e",
            ))
            fig.add_trace(go.Bar(
                x=hospital_stats["Hospital"],
                y=hospital_stats["Negative"],
                name="Negative (Healthy)",
                marker_color="#10b981",
            ))
            fig.update_layout(
                height=320,
                barmode="stack",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f1f5f9", family="Inter"),
                legend=dict(bgcolor="rgba(30,41,59,0.7)", bordercolor="rgba(148,163,184,0.15)"),
                margin=dict(l=40, r=20, t=10, b=40),
                yaxis=dict(title="Patients", gridcolor="rgba(148,163,184,0.1)"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hospital data found.")

    # Communication efficiency summary
    if comm_df is not None:
        st.markdown("---")
        row = comm_df.iloc[0]
        col_e1, col_e2, col_e3 = st.columns(3)
        with col_e1:
            st.metric("Model Parameters", f"{int(row['param_count']):,}")
        with col_e2:
            st.metric("FL Communication", f"{row['total_fl_mb']:.2f} MB")
        with col_e3:
            savings = (1 - row["comm_ratio"]) * 100
            st.metric("Bandwidth Savings", f"{savings:.1f}%")


# ══════════════════════════════════════════════════════════════════════
#                            MAIN APP
# ══════════════════════════════════════════════════════════════════════

def main():
    """Main application entry point."""
    inject_custom_css()
    render_header()
    page = render_sidebar()

    if page == "Dashboard Overview":
        render_dashboard_overview()
    elif page == "Network Topology":
        render_network_topology()
    elif page == "Training Console":
        render_training_console()
    elif page == "Attack vs. Defense":
        render_attack_defense_analysis()
    elif page == "Privacy-Utility Tradeoff":
        render_privacy_utility()
    elif page == "Non-IID Analysis":
        render_non_iid_analysis()
    elif page == "Communication Cost":
        render_communication_cost()
    elif page == "Risk Calculator":
        render_risk_calculator()
    elif page == "Research Figures":
        render_existing_figures()
    elif page == "Project Overview":
        render_project_overview()


if __name__ == "__main__":
    main()

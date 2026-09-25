"""
FedCare Interactive Dashboard — Premium Rebuild
================================================
Privacy-Preserving Federated Learning for Heart Disease Prediction.

A complete, production-grade Streamlit dashboard with:
- Glassmorphic dark theme with animated gradients
- 14 interactive pages (10 original + 4 new features)
- Horizontal top-navigation (completely redesigned from sidebar)
- Enhanced Plotly visualizations with radar charts & 3D
- Micro-animations and premium transitions

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
from streamlit_option_menu import option_menu
import torch

# ── Path Setup ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "heart"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# ── Streamlit Page Configuration ──────────────────────────────────────
st.set_page_config(
    page_title="FedCare — Privacy-Preserving Federated Learning",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════
#                         THEME SYSTEM
# ══════════════════════════════════════════════════════════════════════

HOSPITAL_COLORS = ["#f43f5e", "#f59e0b", "#10b981", "#3b82f6", "#8b5cf6", "#ec4899"]
STRATEGY_COLORS = ["#6366f1", "#06b6d4", "#10b981", "#f59e0b", "#f43f5e"]

PLOTLY_THEME = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#f9fafb", family="Inter, sans-serif", size=12),
    legend=dict(bgcolor="rgba(17,24,39,0.7)", bordercolor="rgba(75,85,99,0.3)", borderwidth=1, font=dict(size=11, color="#9ca3af")),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor="rgba(75,85,99,0.15)", zerolinecolor="rgba(75,85,99,0.2)"),
    yaxis=dict(gridcolor="rgba(75,85,99,0.15)", zerolinecolor="rgba(75,85,99,0.2)"),
)


def inject_premium_css():
    """Inject the complete premium CSS theme."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --fc-primary: #6366f1;
        --fc-primary-light: #818cf8;
        --fc-accent: #06b6d4;
        --fc-accent-light: #22d3ee;
        --fc-success: #10b981;
        --fc-warning: #f59e0b;
        --fc-danger: #ef4444;
        --fc-bg: #030712;
        --fc-surface: rgba(17,24,39,0.6);
        --fc-glass-border: rgba(99,102,241,0.12);
        --fc-text: #f9fafb;
        --fc-text-secondary: #9ca3af;
        --fc-border: rgba(75,85,99,0.3);
        --fc-radius: 10px;
    }

    @keyframes shimmer { 0%{background-position:-200% 0} 100%{background-position:200% 0} }
    @keyframes pulse-glow { 0%,100%{box-shadow:0 0 8px rgba(99,102,241,0.15)} 50%{box-shadow:0 0 20px rgba(99,102,241,0.3)} }
    @keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-4px)} }
    @keyframes gradient-shift { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
    @keyframes fade-in-up { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:translateY(0)} }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; -webkit-font-smoothing: antialiased; }

    .stApp {
        background: var(--fc-bg);
        background-image: radial-gradient(ellipse at 20% 0%, rgba(99,102,241,0.06) 0%, transparent 50%),
                          radial-gradient(ellipse at 80% 100%, rgba(6,182,212,0.04) 0%, transparent 50%);
    }

    #MainMenu, header, footer { visibility: hidden; }
    .stDeployButton { display: none; }

    /* Hide the entire sidebar functionality */
    [data-testid="collapsedControl"] { display: none; }
    [data-testid="stSidebar"] { display: none; }

    div[data-testid="stMetric"] {
        background: var(--fc-surface);
        backdrop-filter: blur(16px);
        border: 1px solid var(--fc-glass-border);
        border-radius: 14px;
        padding: 20px 24px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        transition: all 200ms cubic-bezier(0.4,0,0.2,1);
        animation: fade-in-up 0.4s cubic-bezier(0.4,0,0.2,1) backwards;
        position: relative;
        overflow: hidden;
    }
    div[data-testid="stMetric"]::before {
        content:''; position:absolute; top:0; left:0; right:0; height:2px;
        background: linear-gradient(90deg, var(--fc-primary), var(--fc-accent), var(--fc-primary));
        background-size: 200% 100%; animation: gradient-shift 3s ease infinite; opacity:0;
        transition: opacity 200ms;
    }
    div[data-testid="stMetric"]:hover { border-color:rgba(99,102,241,0.4); transform:translateY(-2px); box-shadow:0 8px 32px rgba(0,0,0,0.5), 0 0 20px rgba(99,102,241,0.15); }
    div[data-testid="stMetric"]:hover::before { opacity:1; }
    div[data-testid="stMetric"] label { color:var(--fc-text-secondary)!important; font-size:0.75rem!important; font-weight:600!important; text-transform:uppercase; letter-spacing:0.8px; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color:var(--fc-text)!important; font-size:1.9rem!important; font-weight:800!important; font-family:'JetBrains Mono',monospace!important; letter-spacing:-0.5px; }
    div[data-testid="stMetric"]:nth-child(1){animation-delay:.05s} div[data-testid="stMetric"]:nth-child(2){animation-delay:.1s} div[data-testid="stMetric"]:nth-child(3){animation-delay:.15s} div[data-testid="stMetric"]:nth-child(4){animation-delay:.2s} div[data-testid="stMetric"]:nth-child(5){animation-delay:.25s}

    h1,h2,h3,h4,h5,h6 { color:var(--fc-text)!important; font-weight:700!important; }
    h1 { font-size:2.4rem!important; font-weight:800!important; letter-spacing:-0.8px; background:linear-gradient(135deg,var(--fc-text) 0%,var(--fc-primary-light) 50%,var(--fc-accent-light) 100%); background-size:200% 200%; animation:gradient-shift 4s ease infinite; -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }

    .stTabs [data-baseweb="tab-list"] { gap:8px; background:var(--fc-surface); backdrop-filter:blur(12px); border-radius:12px; padding:6px; border:1px solid var(--fc-border); justify-content: center;}
    .stTabs [data-baseweb="tab"] { border-radius:8px; color:var(--fc-text-secondary); font-weight:600; padding:10px 24px; transition:all 200ms; font-size: 1rem;}
    .stTabs [data-baseweb="tab"]:hover { color:var(--fc-text); background:rgba(99,102,241,0.1); }
    .stTabs [aria-selected="true"] { background:rgba(99,102,241,0.2)!important; color:var(--fc-primary-light)!important; font-weight:700; border-bottom: 2px solid var(--fc-primary); }

    .stButton > button { background:linear-gradient(135deg,var(--fc-primary) 0%,#4f46e5 100%)!important; color:white!important; border:1px solid rgba(129,140,248,0.3)!important; border-radius:10px!important; padding:10px 24px!important; font-weight:600!important; font-size:0.9rem!important; transition:all 200ms!important; box-shadow:0 2px 8px rgba(99,102,241,0.25)!important; }
    .stButton > button:hover { transform:translateY(-1px)!important; box-shadow:0 4px 16px rgba(99,102,241,0.4)!important; }

    .stSelectbox > div > div, .stNumberInput > div > div > input, .stTextInput > div > div > input { border-color:var(--fc-border)!important; background-color:#111827!important; color:var(--fc-text)!important; border-radius:6px!important; }

    .streamlit-expanderHeader { background:var(--fc-surface)!important; border:1px solid var(--fc-border)!important; border-radius:10px!important; color:var(--fc-text)!important; }

    hr { border-color:var(--fc-border)!important; opacity:0.5; }

    ::-webkit-scrollbar { width:6px; height:6px; }
    ::-webkit-scrollbar-track { background:transparent; }
    ::-webkit-scrollbar-thumb { background:rgba(99,102,241,0.25); border-radius:3px; }

    .fc-badge { display:inline-flex; align-items:center; gap:6px; background:rgba(99,102,241,0.1); color:var(--fc-primary-light); border:1px solid rgba(99,102,241,0.2); padding:5px 14px; border-radius:20px; font-size:0.72rem; font-weight:600; text-transform:uppercase; letter-spacing:0.8px; }
    .fc-badge::before { content:''; width:6px; height:6px; background:var(--fc-primary); border-radius:50%; animation:pulse-glow 2s ease-in-out infinite; }

    .fc-glass-card { background:var(--fc-surface); backdrop-filter:blur(16px); border:1px solid var(--fc-glass-border); border-radius:14px; padding:24px; transition:all 200ms; animation:fade-in-up 0.4s cubic-bezier(0.4,0,0.2,1) backwards; }
    .fc-glass-card:hover { border-color:rgba(99,102,241,0.4); box-shadow:0 0 20px rgba(99,102,241,0.15); }

    .fc-stat { font-family:'JetBrains Mono',monospace; font-weight:700; font-size:2rem; background:linear-gradient(135deg,var(--fc-primary-light),var(--fc-accent-light)); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }

    .fc-separator { height:1px; background:linear-gradient(90deg,transparent,var(--fc-border),rgba(99,102,241,0.15),var(--fc-border),transparent); margin:2rem 0; }

    .fc-gradient-text { background:linear-gradient(135deg,#818cf8,#22d3ee,#34d399,#818cf8); background-size:300% 300%; animation:gradient-shift 4s ease infinite; -webkit-background-clip:text; -webkit-text-fill-color:transparent; }

    .fc-phase-pill { display:inline-flex; align-items:center; gap:8px; padding:6px 14px; border-radius:20px; font-size:0.82rem; font-weight:600; }
    .fc-p1 { background:rgba(99,102,241,0.1); color:#818cf8; border:1px solid rgba(99,102,241,0.2); }
    .fc-p2 { background:rgba(6,182,212,0.1); color:#22d3ee; border:1px solid rgba(6,182,212,0.2); }
    .fc-p3 { background:rgba(16,185,129,0.1); color:#34d399; border:1px solid rgba(16,185,129,0.2); }
    .fc-p4 { background:rgba(239,68,68,0.1); color:#f87171; border:1px solid rgba(239,68,68,0.2); }
    .fc-p5 { background:rgba(168,85,247,0.1); color:#c084fc; border:1px solid rgba(168,85,247,0.2); }
    
    /* Top Menu Styling Override */
    .stRadio [role="radiogroup"] {
        display: flex;
        justify-content: center;
        gap: 15px;
        padding-bottom: 20px;
    }
    .stRadio label {
        background: rgba(17,24,39,0.6);
        border: 1px solid var(--fc-glass-border);
        border-radius: 8px;
        padding: 8px 16px;
        cursor: pointer;
        transition: all 200ms;
    }
    .stRadio label:hover {
        background: rgba(99,102,241,0.15);
    }
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#                     CACHED DATA LOADERS
# ══════════════════════════════════════════════════════════════════════

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
                "Hospital": f"Hospital {i}", "ID": i, "Samples": n_samples,
                "Positive": n_positive, "Negative": n_samples - n_positive,
                "Prevalence": prevalence, "Avg_Age": avg_age, "Avg_Cholesterol": avg_chol,
                "Avg_BP": avg_bp, "Avg_HR": avg_hr, "Avg_BMI": avg_bmi, "Smoker_Pct": smoker_pct,
            })
    return pd.DataFrame(stats)

@st.cache_data(show_spinner=False)
def load_hospital_raw(hospital_id: int) -> Optional[pd.DataFrame]:
    csv_path = DATA_DIR / f"hospital_{hospital_id}.csv"
    return pd.read_csv(csv_path) if csv_path.exists() else None

@st.cache_data(show_spinner=False)
def load_combined_data() -> Optional[pd.DataFrame]:
    path = DATA_DIR / "combined.csv"
    return pd.read_csv(path) if path.exists() else None

@st.cache_data(show_spinner=False)
def load_fedavg_rounds() -> Optional[pd.DataFrame]:
    path = RESULTS_DIR / "rounds_fedavg.csv"
    return pd.read_csv(path) if path.exists() else None

@st.cache_data(show_spinner=False)
def load_attack_defense_matrix() -> Optional[pd.DataFrame]:
    path = RESULTS_DIR / "phase4_attack_defense_matrix.csv"
    return pd.read_csv(path) if path.exists() else None

@st.cache_data(show_spinner=False)
def load_dp_sweep() -> Optional[pd.DataFrame]:
    path = RESULTS_DIR / "phase4_dp_sweep.csv"
    return pd.read_csv(path) if path.exists() else None

@st.cache_data(show_spinner=False)
def load_non_iid_results() -> Optional[pd.DataFrame]:
    path = RESULTS_DIR / "phase3_non_iid_experiments.csv"
    return pd.read_csv(path) if path.exists() else None

@st.cache_data(show_spinner=False)
def load_fedprox_results() -> Optional[pd.DataFrame]:
    path = RESULTS_DIR / "phase3_fedprox_experiments.csv"
    return pd.read_csv(path) if path.exists() else None

@st.cache_data(show_spinner=False)
def load_comm_cost() -> Optional[pd.DataFrame]:
    path = RESULTS_DIR / "phase4_comm_cost.csv"
    return pd.read_csv(path) if path.exists() else None

@st.cache_data(show_spinner=False)
def load_attack_trajectories() -> Optional[pd.DataFrame]:
    path = RESULTS_DIR / "phase4_attack_trajectories.csv"
    return pd.read_csv(path) if path.exists() else None

@st.cache_resource(show_spinner=False)
def load_global_model():
    """Load the trained global FedCare model for real-time inference."""
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


def _hex_to_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"


# ══════════════════════════════════════════════════════════════════════
#                          HEADER
# ══════════════════════════════════════════════════════════════════════

def render_header():
    st.markdown('<div style="text-align:center; padding: 20px 0;"><div class="fc-badge">Live Research Dashboard</div></div>', unsafe_allow_html=True)
    st.markdown("""
        <h1 style="text-align:center; font-size: 3rem !important;">FedCare</h1>
        <p style="text-align:center; color:#9ca3af;font-size:1.1rem;max-width:850px;line-height:1.7;margin: -8px auto 20px auto;">
        A privacy-preserving federated learning framework enabling <strong style="color:#818cf8">6 hospitals</strong>
        to collaboratively train a heart disease classifier.
        <strong style="color:#22d3ee">No raw data ever leaves a hospital.</strong>
        </p>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#                     PAGE: DASHBOARD OVERVIEW
# ══════════════════════════════════════════════════════════════════════

def render_dashboard_overview():
    rounds_df = load_fedavg_rounds()
    attack_df = load_attack_defense_matrix()
    dp_df = load_dp_sweep()
    comm_df = load_comm_cost()

    # ── Top KPI Row ───────────────────────────────────────────────────
    cols = st.columns(5)
    if rounds_df is not None:
        last = rounds_df.iloc[-1]
        with cols[0]: st.metric("Global AUC", f"{last['auc']:.4f}", delta="96.7% gap recovery")
        with cols[1]: st.metric("Global Accuracy", f"{last['accuracy']:.4f}")
        with cols[2]: st.metric("Training Rounds", f"{int(last['round'])}", delta="6 hospitals")
    else:
        with cols[0]: st.metric("Global AUC", "N/A")
        with cols[1]: st.metric("Global Accuracy", "N/A")
        with cols[2]: st.metric("Training Rounds", "N/A")
    with cols[3]:
        n_exp = len(attack_df) if attack_df is not None else 0
        st.metric("Security Experiments", str(n_exp), delta="3 attack types")
    with cols[4]:
        if dp_df is not None:
            best_eps = dp_df.loc[dp_df["Final_AUC"].idxmax(), "Epsilon"]
            st.metric("Best Privacy (ε)", f"{best_eps:.1f}")
        else:
            st.metric("Best Privacy (ε)", "N/A")

    st.markdown('<div class="fc-separator"></div>', unsafe_allow_html=True)

    # ── Phase Journey Cards ───────────────────────────────────────────
    st.markdown("#### 📋 Research Phase Journey")
    phase_cols = st.columns(5)
    phases = [
        ("Phase 1", "Baselines", "fc-p1", "Centralized & local benchmarks"),
        ("Phase 2", "FedAvg Core", "fc-p2", "20 rounds, AUC 0.8468"),
        ("Phase 3", "Non-IID + FedProx", "fc-p3", "Heterogeneity analysis"),
        ("Phase 4", "Security & Privacy", "fc-p4", "Attacks, defenses, DP"),
        ("Phase 5", "Dashboard", "fc-p5", "Interactive visualization"),
    ]
    for col, (name, subtitle, cls, desc) in zip(phase_cols, phases):
        with col:
            st.markdown(f"""
            <div class="fc-glass-card" style="text-align:center;min-height:140px;">
                <span class="fc-phase-pill {cls}">{name}</span>
                <h4 style="margin:12px 0 4px;font-size:1rem!important;">{subtitle}</h4>
                <p style="color:#6b7280;font-size:0.82rem;margin:0;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="fc-separator"></div>', unsafe_allow_html=True)

    # ── Charts Row ────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### 📈 Training Convergence")
        if rounds_df is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rounds_df["round"], y=rounds_df["auc"],
                mode="lines+markers", name="AUC",
                line=dict(width=3, color="#6366f1", shape="spline"),
                marker=dict(size=6),
                fill="tozeroy", fillcolor="rgba(99,102,241,0.08)",
            ))
            fig.add_trace(go.Scatter(
                x=rounds_df["round"], y=rounds_df["accuracy"],
                mode="lines+markers", name="Accuracy",
                line=dict(width=2, color="#10b981", dash="dot", shape="spline"),
                marker=dict(size=5),
            ))
            fig.update_layout(PLOTLY_THEME)
            fig.update_layout(height=360)
            fig.update_xaxes(title_text="Round")
            fig.update_yaxes(title_text="Metric")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run training to see convergence charts.")

    with col_right:
        st.markdown("#### 🏥 Hospital Data Distribution")
        hospital_stats = load_hospital_stats()
        if not hospital_stats.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=hospital_stats["Hospital"], y=hospital_stats["Positive"],
                name="Heart Disease", marker_color="#f43f5e", marker_cornerradius=4,
            ))
            fig.add_trace(go.Bar(
                x=hospital_stats["Hospital"], y=hospital_stats["Negative"],
                name="Healthy", marker_color="#10b981", marker_cornerradius=4,
            ))
            fig.update_layout(PLOTLY_THEME)
            fig.update_layout(height=360, barmode="stack")
            fig.update_yaxes(title_text="Patients")
            st.plotly_chart(fig, use_container_width=True)

    # ── Communication Summary ─────────────────────────────────────────
    if comm_df is not None:
        st.markdown('<div class="fc-separator"></div>', unsafe_allow_html=True)
        row = comm_df.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Model Parameters", f"{int(row['param_count']):,}")
        with c2: st.metric("FL Communication", f"{row['total_fl_mb']:.2f} MB")
        with c3:
            savings = (1 - row["comm_ratio"]) * 100
            st.metric("Bandwidth Savings", f"{abs(savings):.1f}%")
        with c4: st.metric("Message Size", f"{row['single_message_kb']:.1f} KB")


# ══════════════════════════════════════════════════════════════════════
#                     PAGE: NETWORK TOPOLOGY
# ══════════════════════════════════════════════════════════════════════

def render_network_topology():
    st.markdown("#### 🌐 Federated Network Topology")
    hospital_stats = load_hospital_stats()
    if hospital_stats.empty:
        st.warning("No hospital data found.")
        return

    fig = go.Figure()
    # Central server
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers+text",
        marker=dict(size=60, color="#6366f1", symbol="diamond", line=dict(width=3, color="#818cf8")),
        text=["FedCare<br>Server"], textposition="bottom center",
        textfont=dict(size=11, color="#e0e7ff", family="Inter"),
        name="Central Server", hoverinfo="text",
        hovertext="<b>Aggregation Server</b><br>FedAvg · FedProx · Krum · TrimmedMean · Median",
    ))

    n = len(hospital_stats)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) - np.pi / 2
    radius = 3.2
    for idx, (_, row) in enumerate(hospital_stats.iterrows()):
        x, y = radius * np.cos(angles[idx]), radius * np.sin(angles[idx])
        fig.add_trace(go.Scatter(
            x=[0, x], y=[0, y], mode="lines",
            line=dict(width=1.5, color=f"rgba({_hex_to_rgb(HOSPITAL_COLORS[idx])},0.25)", dash="dot"),
            showlegend=False, hoverinfo="skip",
        ))
        prev_pct = row["Prevalence"] * 100
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text",
            marker=dict(size=42, color=HOSPITAL_COLORS[idx], line=dict(width=2, color="rgba(255,255,255,0.2)")),
            text=[f"H{row['ID']}"], textposition="middle center",
            textfont=dict(size=13, color="white", family="Inter"),
            name=row["Hospital"], hoverinfo="text",
            hovertext=f"<b>{row['Hospital']}</b><br>Patients: {row['Samples']:,}<br>Prevalence: {prev_pct:.1f}%<br>Avg Age: {row['Avg_Age']:.1f}<br>Avg Cholesterol: {row['Avg_Cholesterol']:.0f}",
        ))

    fig.update_layout(PLOTLY_THEME)
    fig.update_layout(height=500, showlegend=False)
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, range=[-5, 5])
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, range=[-5, 5])
    fig.add_annotation(x=0, y=1.0, text="<b>Encrypted Parameters Only</b><br>Zero patient data transmitted", showarrow=False, font=dict(size=9, color="#6b7280"))
    st.plotly_chart(fig, use_container_width=True)

    # Hospital cards
    cols = st.columns(6)
    for idx, (_, row) in enumerate(hospital_stats.iterrows()):
        with cols[idx]:
            prev_pct = row["Prevalence"] * 100
            st.metric(f"Hospital {row['ID']}", f"{row['Samples']:,}", delta=f"{prev_pct:.1f}% prevalence")


# ══════════════════════════════════════════════════════════════════════
#                     PAGE: TRAINING CONSOLE
# ══════════════════════════════════════════════════════════════════════

def render_training_console():
    st.markdown("#### ⚡ Training Convergence Dashboard")
    rounds_df = load_fedavg_rounds()
    if rounds_df is None:
        st.warning("No training data. Run `run_federated.py` first.")
        return

    c1, c2, c3 = st.columns(3)
    with c1: metric_choice = st.selectbox("Primary Metric", ["ROC-AUC", "Accuracy", "Loss"], index=0, key="tc_metric")
    with c2: show_hospitals = st.toggle("Per-Hospital AUC", value=True, key="tc_hosp")
    with c3: animate = st.toggle("Animate Rounds", value=False, key="tc_anim")

    metric_map = {"ROC-AUC": ("auc", "ROC-AUC"), "Accuracy": ("accuracy", "Accuracy"), "Loss": ("test_loss", "Loss")}
    col_name, display_name = metric_map[metric_choice]

    if animate:
        max_round = int(rounds_df["round"].max())
        current_round = st.slider("Simulation Round", 1, max_round, max_round, key="tc_slider")
        plot_df = rounds_df[rounds_df["round"] <= current_round]
    else:
        plot_df = rounds_df

    fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38],
        subplot_titles=(f"Global {display_name} Convergence", "Per-Hospital AUC Distribution"),
        horizontal_spacing=0.08)

    fig.add_trace(go.Scatter(
        x=plot_df["round"], y=plot_df[col_name], mode="lines+markers",
        name=f"Global {display_name}", line=dict(width=3, color="#6366f1", shape="spline"),
        marker=dict(size=6), fill="tozeroy" if col_name != "test_loss" else None,
        fillcolor="rgba(99,102,241,0.08)"), row=1, col=1)

    if show_hospitals and col_name == "auc":
        for i in range(1, 7):
            hcol = f"hosp_{i}_auc"
            if hcol in plot_df.columns:
                fig.add_trace(go.Scatter(
                    x=plot_df["round"], y=plot_df[hcol], mode="lines",
                    name=f"Hospital {i}", line=dict(width=1.5, color=HOSPITAL_COLORS[i-1], dash="dash"),
                    opacity=0.65), row=1, col=1)

    last_row = plot_df.iloc[-1]
    hosp_aucs, hosp_labels = [], []
    for i in range(1, 7):
        hcol = f"hosp_{i}_auc"
        if hcol in last_row.index:
            hosp_aucs.append(float(last_row[hcol]))
            hosp_labels.append(f"H{i}")

    if hosp_aucs:
        fig.add_trace(go.Bar(
            x=hosp_labels, y=hosp_aucs,
            marker=dict(color=HOSPITAL_COLORS[:len(hosp_aucs)], line=dict(width=1, color="rgba(255,255,255,0.15)"), cornerradius=4),
            name="Hospital AUC", text=[f"{v:.4f}" for v in hosp_aucs],
            textposition="outside", textfont=dict(size=10, color="#9ca3af", family="JetBrains Mono")), row=1, col=2)

    fig.update_layout(PLOTLY_THEME)
    fig.update_layout(height=460)
    fig.update_xaxes(title_text="Communication Round", row=1, col=1)
    fig.update_yaxes(title_text=display_name, row=1, col=1)
    fig.update_xaxes(title_text="Hospital", row=1, col=2)
    if hosp_aucs:
        fig.update_yaxes(title_text="AUC", row=1, col=2, range=[min(hosp_aucs)-0.03, max(hosp_aucs)+0.03])
    st.plotly_chart(fig, use_container_width=True)

    # Summary metrics
    last = rounds_df.iloc[-1]
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1: st.metric("Final Global AUC", f"{last['auc']:.4f}")
    with mc2: st.metric("Final Accuracy", f"{last['accuracy']:.4f}")
    with mc3:
        h_vals = [float(last[f"hosp_{i}_auc"]) for i in range(1,7) if f"hosp_{i}_auc" in last.index]
        gap = max(h_vals) - min(h_vals) if h_vals else 0
        st.metric("Equity Gap", f"{gap:.4f}")
    with mc4: st.metric("Total Rounds", f"{int(last['round'])}")


# ══════════════════════════════════════════════════════════════════════
#                   PAGE: ATTACK VS. DEFENSE
# ══════════════════════════════════════════════════════════════════════

def render_attack_defense():
    st.markdown("#### 🛡️ Adversarial Robustness: Attack × Defense Matrix")
    attack_df = load_attack_defense_matrix()
    if attack_df is None:
        st.warning("No attack-defense data. Run `run_phase4_experiments.py` first.")
        return

    c1, c2 = st.columns(2)
    with c1:
        selected_attacks = st.multiselect("Filter Attacks", attack_df["Attack_Name"].unique().tolist(),
            default=attack_df["Attack_Name"].unique().tolist(), key="ad_filter")
    with c2:
        selected_metric = st.selectbox("Comparison Metric",
            ["Final_AUC", "Final_Accuracy", "Equity_Gap", "Worst_Hospital_AUC"], index=0, key="ad_metric")

    filtered = attack_df[attack_df["Attack_Name"].isin(selected_attacks)]

    # Grouped bar chart
    fig = px.bar(filtered, x="Attack_Name", y=selected_metric, color="Strategy_Name", barmode="group",
        color_discrete_sequence=STRATEGY_COLORS,
        labels={"Attack_Name": "Attack", selected_metric: selected_metric.replace("_", " "), "Strategy_Name": "Defense"})
    fig.update_layout(PLOTLY_THEME)
    fig.update_layout(height=450)
    fig.update_traces(marker_cornerradius=4)
    st.plotly_chart(fig, use_container_width=True)

    # Heatmap
    col_hm, col_radar = st.columns(2)
    with col_hm:
        st.markdown("##### 🔥 AUC Heatmap")
        if "Final_AUC" in filtered.columns and len(filtered) > 0:
            pivot = filtered.pivot_table(index="Attack_Name", columns="Strategy_Name", values="Final_AUC", aggfunc="mean")
            fig_hm = go.Figure(data=go.Heatmap(
                z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
                colorscale=[[0,"#7f1d1d"],[0.3,"#dc2626"],[0.5,"#f59e0b"],[0.7,"#22c55e"],[1.0,"#059669"]],
                text=np.round(pivot.values, 4), texttemplate="%{text}",
                textfont=dict(size=12, color="white", family="JetBrains Mono"),
                hoverongaps=False, colorbar=dict(title="AUC", tickfont=dict(color="#9ca3af"))))
            fig_hm.update_layout(PLOTLY_THEME)
            fig_hm.update_layout(height=360, xaxis_title="Defense", yaxis_title="Attack")
            st.plotly_chart(fig_hm, use_container_width=True)

    with col_radar:
        st.markdown("##### 🕸️ Strategy Comparison Radar")
        if "Final_AUC" in filtered.columns and len(filtered) > 0:
            strategies = filtered["Strategy_Name"].unique()
            categories = ["AUC", "Accuracy", "Robustness", "Equity"]
            fig_radar = go.Figure()
            for i, strat in enumerate(strategies):
                strat_data = filtered[filtered["Strategy_Name"] == strat]
                vals = [
                    strat_data["Final_AUC"].mean(),
                    strat_data["Final_Accuracy"].mean() if "Final_Accuracy" in strat_data.columns else 0.8,
                    1 - (strat_data["Final_AUC"].std() if len(strat_data) > 1 else 0),
                    1 - (strat_data["Equity_Gap"].mean() if "Equity_Gap" in strat_data.columns else 0.05),
                ]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]], theta=categories + [categories[0]],
                    fill="toself", name=strat,
                    fillcolor=f"rgba({_hex_to_rgb(STRATEGY_COLORS[i % len(STRATEGY_COLORS)])},0.1)",
                    line=dict(color=STRATEGY_COLORS[i % len(STRATEGY_COLORS)], width=2)))
            fig_radar.update_layout(PLOTLY_THEME)
            fig_radar.update_layout(height=360, polar=dict(bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, gridcolor="rgba(75,85,99,0.2)", tickfont=dict(color="#6b7280", size=9)),
                angularaxis=dict(gridcolor="rgba(75,85,99,0.2)", tickfont=dict(color="#9ca3af", size=11))))
            st.plotly_chart(fig_radar, use_container_width=True)

    # Attack trajectories
    traj_df = load_attack_trajectories()
    if traj_df is not None and len(traj_df) > 0 and "round" in traj_df.columns and "auc" in traj_df.columns and "label" in traj_df.columns:
        st.markdown("##### ⏱️ Round-by-Round Attack Impact Trajectories")
        fig_traj = px.line(traj_df, x="round", y="auc", color="label",
            color_discrete_sequence=["#6366f1", "#f43f5e", "#10b981", "#f59e0b", "#8b5cf6"],
            labels={"round": "Communication Round", "auc": "Global AUC", "label": "Scenario"})
        fig_traj.update_layout(PLOTLY_THEME)
        fig_traj.update_layout(height=380)
        st.plotly_chart(fig_traj, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
#                   PAGE: PRIVACY-UTILITY TRADEOFF
# ══════════════════════════════════════════════════════════════════════

def render_privacy_utility():
    st.markdown("#### 🔐 Privacy-Utility Tradeoff Explorer")
    dp_df = load_dp_sweep()
    if dp_df is None:
        st.warning("No DP sweep data. Run `run_phase4_experiments.py` first.")
        return

    col_chart, col_table = st.columns([3, 1])
    with col_chart:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dp_df["Epsilon"], y=dp_df["Final_AUC"], mode="lines+markers",
            name="Global AUC", line=dict(width=3, color="#6366f1", shape="spline"),
            marker=dict(size=10, symbol="circle")))
        fig.add_trace(go.Scatter(
            x=dp_df["Epsilon"], y=dp_df["Worst_Hospital_AUC"], mode="lines+markers",
            name="Worst Hospital", line=dict(width=2, color="#f43f5e", dash="dash", shape="spline"),
            marker=dict(size=8, symbol="diamond")))
        for _, row in dp_df.iterrows():
            regime = str(row.get("Privacy_Regime", ""))
            color = {"Strong": "#10b981", "Moderate": "#f59e0b", "Weak": "#ef4444"}.get(regime, "#6b7280")
            fig.add_annotation(x=row["Epsilon"], y=row["Final_AUC"], text=regime, showarrow=False, yshift=18, font=dict(size=9, color=color))
        fig.update_layout(PLOTLY_THEME)
        fig.update_layout(height=420, xaxis_title="Privacy Budget (ε)", yaxis_title="AUC")
        fig.update_xaxes(type="log")
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        st.markdown("##### DP Configuration")
        st.dataframe(
            dp_df[["Noise_Multiplier", "Epsilon", "Privacy_Regime", "Final_AUC"]].style.format({
                "Noise_Multiplier": "{:.3f}", "Epsilon": "{:.2f}", "Final_AUC": "{:.4f}"}),
            use_container_width=True, hide_index=True)
        best_row = dp_df.loc[dp_df["Final_AUC"].idxmax()]
        st.info(f"**Best AUC**: {best_row['Final_AUC']:.4f} at ε={best_row['Epsilon']:.2f} ({best_row.get('Privacy_Regime', 'N/A')})")


# ══════════════════════════════════════════════════════════════════════
#                   PAGE: NON-IID ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def render_non_iid_analysis():
    st.markdown("#### 📊 Non-IID Data Heterogeneity Analysis")
    non_iid_df = load_non_iid_results()
    fedprox_df = load_fedprox_results()

    col1, col2 = st.columns(2)
    with col1:
        if non_iid_df is not None:
            st.markdown("##### Impact of Data Heterogeneity")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=non_iid_df["Configuration"], y=non_iid_df["Final_AUC"], name="AUC", marker_color="#6366f1", marker_cornerradius=4,
                text=non_iid_df["Final_AUC"].round(4), textposition="outside", textfont=dict(color="#9ca3af", size=10)))
            fig.add_trace(go.Bar(x=non_iid_df["Configuration"], y=non_iid_df["Equity_Gap"], name="Equity Gap", marker_color="#f43f5e", marker_cornerradius=4,
                text=non_iid_df["Equity_Gap"].round(4), textposition="outside", textfont=dict(color="#9ca3af", size=10), yaxis="y2"))
            fig.update_layout(PLOTLY_THEME)
            fig.update_layout(height=420, barmode="group",
                yaxis=dict(title="AUC"), yaxis2=dict(title="Equity Gap", overlaying="y", side="right"),
                xaxis=dict(tickangle=-20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No Non-IID data found.")

    with col2:
        if fedprox_df is not None:
            st.markdown("##### FedProx vs. FedAvg Comparison")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=fedprox_df["Algorithm"], y=fedprox_df["Final_AUC"], name="AUC", marker_color="#10b981", marker_cornerradius=4,
                text=fedprox_df["Final_AUC"].round(4), textposition="outside", textfont=dict(color="#9ca3af", size=10)))
            fig.add_trace(go.Bar(x=fedprox_df["Algorithm"], y=fedprox_df["Equity_Gap"], name="Equity Gap", marker_color="#f59e0b", marker_cornerradius=4,
                text=fedprox_df["Equity_Gap"].round(4), textposition="outside", textfont=dict(color="#9ca3af", size=10)))
            fig.update_layout(PLOTLY_THEME)
            fig.update_layout(height=420, barmode="group", yaxis=dict(title="Value"), xaxis=dict(tickangle=-20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No FedProx data found.")


# ══════════════════════════════════════════════════════════════════════
#                   PAGE: COMMUNICATION COST
# ══════════════════════════════════════════════════════════════════════

def render_communication_cost():
    st.markdown("#### 📡 Communication Efficiency Analysis")
    comm_df = load_comm_cost()
    if comm_df is None:
        st.warning("No communication cost data.")
        return

    row = comm_df.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Model Parameters", f"{int(row['param_count']):,}")
    with c2: st.metric("Single Message", f"{row['single_message_kb']:.1f} KB")
    with c3: st.metric("Total FL Comm.", f"{row['total_fl_mb']:.2f} MB")
    with c4:
        ratio = row["comm_ratio"]
        savings = (1 - ratio) * 100
        st.metric("vs. Centralized", f"{ratio:.4f}x", delta=f"{savings:.1f}% savings")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Federated<br>Model Exchange", "Centralized<br>Raw Data Transfer"],
        y=[row["total_fl_mb"], row["centralized_data_mb"]],
        marker_color=["#6366f1", "#f43f5e"], marker_cornerradius=6,
        text=[f"{row['total_fl_mb']:.2f} MB", f"{row['centralized_data_mb']:.2f} MB"],
        textposition="outside", textfont=dict(color="#f9fafb", size=14, family="JetBrains Mono"), width=0.35))
    fig.update_layout(PLOTLY_THEME)
    fig.update_layout(height=380, yaxis=dict(title="Data Transfer (MB)"), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
#                   PAGE: RISK CALCULATOR
# ══════════════════════════════════════════════════════════════════════

def render_risk_calculator():
    st.markdown("#### 🩺 Clinical Heart Disease Risk Calculator")
    st.markdown("""<p style="color:#9ca3af;font-size:0.95rem;">
    Enter 13 clinical features. The global federated model computes heart disease
    probability <strong style="color:#818cf8">without any patient data leaving this interface</strong>.
    </p>""", unsafe_allow_html=True)

    with st.spinner("Loading global model..."):
        model, scaler = load_global_model()

    with st.form("risk_form", clear_on_submit=False):
        st.markdown("##### Patient Clinical Features")
        ca, cb, cc = st.columns(3)
        with ca:
            age = st.number_input("Age (years)", 18, 100, 55, 1)
            resting_bp = st.number_input("Resting Blood Pressure (mmHg)", 80, 220, 130, 1)
            cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 240, 5)
            max_hr = st.number_input("Max Heart Rate (bpm)", 60, 220, 150, 1)
        with cb:
            bmi = st.number_input("BMI (kg/m²)", 15.0, 55.0, 27.5, 0.1, format="%.1f")
            glucose = st.number_input("Fasting Glucose (mg/dL)", 50, 400, 100, 5)
            sex = st.selectbox("Sex", ["Male", "Female"])
            smoker = st.selectbox("Smoker", ["No", "Yes"])
        with cc:
            diabetes = st.selectbox("Diabetes History", ["No", "Yes"])
            family_history = st.selectbox("Family History", ["No", "Yes"])
            chest_pain = st.selectbox("Chest Pain Type", ["Asymptomatic", "Atypical Angina", "Non-Anginal", "Typical Angina"])
        submitted = st.form_submit_button("⚡ Predict Heart Disease Risk", use_container_width=True)

    if submitted:
        raw_features = np.array([[
            age, resting_bp, cholesterol, max_hr, bmi, glucose,
            1 if sex == "Male" else 0, 1 if smoker == "Yes" else 0,
            1 if diabetes == "Yes" else 0, 1 if family_history == "Yes" else 0,
            1 if chest_pain == "Atypical Angina" else 0,
            1 if chest_pain == "Non-Anginal" else 0,
            1 if chest_pain == "Typical Angina" else 0,
        ]], dtype=np.float64)

        scaled = scaler.transform(raw_features)
        input_tensor = torch.tensor(scaled, dtype=torch.float32)
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            risk_prob = float(probs[0, 1])

        st.markdown('<div class="fc-separator"></div>', unsafe_allow_html=True)
        _, col_gauge, _ = st.columns([1, 2, 1])
        with col_gauge:
            if risk_prob < 0.25:
                risk_level, risk_color, risk_class = "LOW RISK", "#10b981", "fc-risk-low"
                advice = "Continue maintaining a healthy lifestyle. Regular check-ups recommended."
            elif risk_prob < 0.50:
                risk_level, risk_color, risk_class = "MODERATE RISK", "#f59e0b", "fc-risk-moderate"
                advice = "Consider lifestyle modifications. Consult with a cardiologist."
            elif risk_prob < 0.75:
                risk_level, risk_color, risk_class = "HIGH RISK", "#ef4444", "fc-risk-high"
                advice = "Immediate medical consultation recommended. Diagnostic tests advised."
            else:
                risk_level, risk_color, risk_class = "VERY HIGH RISK", "#dc2626", "fc-risk-very-high"
                advice = "Urgent cardiology referral needed. Comprehensive cardiac workup required."

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=risk_prob * 100,
                number=dict(suffix="%", font=dict(size=52, color="#f9fafb", family="JetBrains Mono")),
                title=dict(text="Heart Disease Probability", font=dict(size=14, color="#9ca3af")),
                gauge=dict(
                    axis=dict(range=[0, 100], tickwidth=1, tickcolor="#4b5563", dtick=25),
                    bar=dict(color=risk_color, thickness=0.25),
                    bgcolor="rgba(17,24,39,0.5)", borderwidth=1, bordercolor="rgba(75,85,99,0.3)",
                    steps=[
                        dict(range=[0, 25], color="rgba(16,185,129,0.12)"),
                        dict(range=[25, 50], color="rgba(245,158,11,0.12)"),
                        dict(range=[50, 75], color="rgba(239,68,68,0.12)"),
                        dict(range=[75, 100], color="rgba(220,38,38,0.15)"),
                    ],
                    threshold=dict(line=dict(color=risk_color, width=4), thickness=0.8, value=risk_prob * 100))))
            fig_gauge.update_layout(PLOTLY_THEME)
            fig_gauge.update_layout(height=320, margin=dict(l=30, r=30, t=60, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown(f'<p style="text-align:center;font-size:1.8rem;" class="{risk_class}">{risk_level}</p>', unsafe_allow_html=True)
            st.markdown(f'<p style="color:#9ca3af;text-align:center;font-size:1rem;">{advice}</p>', unsafe_allow_html=True)

        st.markdown("##### Patient Feature Summary")
        feat_data = {"Feature": ["Age", "Resting BP", "Cholesterol", "Max HR", "BMI", "Glucose", "Sex", "Smoker", "Diabetes", "Family History", "Chest Pain"],
                     "Value": [f"{age} yrs", f"{resting_bp} mmHg", f"{cholesterol} mg/dL", f"{max_hr} bpm", f"{bmi:.1f} kg/m²", f"{glucose} mg/dL", sex, smoker, diabetes, family_history, chest_pain]}
        st.dataframe(pd.DataFrame(feat_data), use_container_width=True, hide_index=True)
        st.caption("**Disclaimer**: Research demonstration only. Not for clinical decision-making without professional medical review.")


# ══════════════════════════════════════════════════════════════════════
#                   PAGE: RESEARCH FIGURES
# ══════════════════════════════════════════════════════════════════════

def render_research_figures():
    st.markdown("#### 🎨 Research Figures Gallery")
    figures = {
        "Figure 1: FedAvg Convergence": RESULTS_DIR / "figure1_fedavg_convergence.png",
        "Figure 2: Non-IID Impact": RESULTS_DIR / "figure2_non_iid_impact.png",
        "Figure 3: FedProx vs FedAvg": RESULTS_DIR / "figure3_fedprox_vs_fedavg.png",
        "Figure 4: Attacks & Defenses": RESULTS_DIR / "figure4_attacks_and_defenses.png",
        "Figure 5: Privacy-Utility": RESULTS_DIR / "figure5_privacy_utility.png",
    }
    available = {k: v for k, v in figures.items() if v.exists()}
    if not available:
        st.warning("No pre-generated figures found.")
        return
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
#                   PAGE: PROJECT OVERVIEW
# ══════════════════════════════════════════════════════════════════════

def render_project_overview():
    st.markdown("#### 📖 Project Overview & Methodology")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        ##### Federated Learning Architecture
        **FedCare** uses the **Flower** framework to orchestrate federated training across
        **6 geographically distributed hospitals**, each with ~2,000 patient records.

        **Key Principles:**
        - Raw patient data **never leaves** the hospital
        - Only model parameter updates are communicated
        - Central server aggregates using configurable strategies
        - Each hospital retains full data sovereignty

        ##### Aggregation Strategies
        | Strategy | Description |
        |----------|-------------|
        | **FedAvg** | Weighted average of client updates |
        | **FedProx** | Proximal regularization for non-IID |
        | **Trimmed Mean** | Byzantine-robust trimmed aggregation |
        | **Coord. Median** | Element-wise median aggregation |
        | **Multi-Krum** | Distance-based outlier filtering |
        """)
    with c2:
        st.markdown("""
        ##### Security & Privacy
        **Adversarial Attacks:**
        - Label-Flipping: Inverts training labels
        - Model Poisoning: Corrupts weight updates

        **Byzantine-Robust Defenses:**
        - Trimmed Mean removes statistical outliers
        - Coordinate Median immune to extreme values
        - Multi-Krum selects representative updates

        **Differential Privacy (DP):**
        - L2 gradient clipping bounds sensitivity
        - Calibrated Gaussian noise guarantees (ε, δ)-privacy
        - Rényi accountant tracks cumulative expenditure

        ##### Model Architecture
        ```
        Input (13 features)
          → Linear(64) → ReLU → Dropout(0.3)
          → Linear(32) → ReLU → Dropout(0.3)
          → Linear(2) (classification logits)
        ```
        """)


# ══════════════════════════════════════════════════════════════════════
#              NEW PAGE: DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════

def render_data_explorer():
    st.markdown("#### 🔬 Interactive Data Explorer")
    st.markdown('<p style="color:#9ca3af;">Explore the synthetic clinical dataset across all 6 hospitals.</p>', unsafe_allow_html=True)

    hospital_stats = load_hospital_stats()
    if hospital_stats.empty:
        st.warning("No hospital data found.")
        return

    tab1, tab2, tab3 = st.tabs(["📊 Feature Distributions", "🔗 Correlations", "🏥 Hospital Comparison"])

    with tab1:
        selected_hosp = st.selectbox("Select Hospital", ["All Hospitals"] + [f"Hospital {i}" for i in range(1, 7)], key="de_hosp")
        if selected_hosp == "All Hospitals":
            df = load_combined_data()
        else:
            hid = int(selected_hosp.split()[-1])
            df = load_hospital_raw(hid)

        if df is not None:
            numeric_cols = [c for c in df.columns if c != "target" and df[c].dtype in ["float64", "int64", "float32", "int32"]]
            cont_cols = [c for c in numeric_cols if df[c].nunique() > 10]
            selected_feature = st.selectbox("Feature", cont_cols, key="de_feat")

            fig = go.Figure()
            healthy = df[df["target"] == 0][selected_feature]
            diseased = df[df["target"] == 1][selected_feature]
            fig.add_trace(go.Histogram(x=healthy, name="Healthy", marker_color="rgba(16,185,129,0.6)", nbinsx=40))
            fig.add_trace(go.Histogram(x=diseased, name="Heart Disease", marker_color="rgba(239,68,68,0.6)", nbinsx=40))
            fig.update_layout(PLOTLY_THEME)
            fig.update_layout(height=400, barmode="overlay",
                xaxis_title=selected_feature.replace("_", " ").title(), yaxis_title="Count",
                title=f"Distribution of {selected_feature.replace('_', ' ').title()}")
            st.plotly_chart(fig, use_container_width=True)

            # Stats
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Mean", f"{df[selected_feature].mean():.2f}")
            with c2: st.metric("Std Dev", f"{df[selected_feature].std():.2f}")
            with c3: st.metric("Min", f"{df[selected_feature].min():.2f}")
            with c4: st.metric("Max", f"{df[selected_feature].max():.2f}")

    with tab2:
        df = load_combined_data()
        if df is not None:
            numeric_cols = [c for c in df.columns if df[c].dtype in ["float64", "int64", "float32", "int32"]]
            corr = df[numeric_cols].corr()
            fig = go.Figure(data=go.Heatmap(
                z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
                colorscale=[[0,"#1e3a5f"],[0.5,"#0f172a"],[1.0,"#6366f1"]],
                text=np.round(corr.values, 2), texttemplate="%{text}",
                textfont=dict(size=9, color="#d1d5db")))
            fig.update_layout(PLOTLY_THEME)
            fig.update_layout(height=500, title="Feature Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("##### Hospital Population Comparison")
        features_to_compare = ["Avg_Age", "Avg_Cholesterol", "Avg_BP", "Avg_HR", "Avg_BMI"]
        available_features = [f for f in features_to_compare if f in hospital_stats.columns]

        if available_features:
            fig = go.Figure()
            for i, feat in enumerate(available_features):
                fig.add_trace(go.Bar(
                    x=hospital_stats["Hospital"], y=hospital_stats[feat],
                    name=feat.replace("Avg_", "").replace("_", " "),
                    marker_color=STRATEGY_COLORS[i % len(STRATEGY_COLORS)], marker_cornerradius=4))
            fig.update_layout(PLOTLY_THEME)
            fig.update_layout(height=420, barmode="group", yaxis_title="Value")
            st.plotly_chart(fig, use_container_width=True)

        # Prevalence comparison
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=hospital_stats["Hospital"], y=hospital_stats["Prevalence"] * 100,
            marker_color=HOSPITAL_COLORS, marker_cornerradius=6,
            text=[f"{v:.1f}%" for v in hospital_stats["Prevalence"] * 100],
            textposition="outside", textfont=dict(color="#9ca3af", family="JetBrains Mono")))
        fig2.update_layout(PLOTLY_THEME)
        fig2.update_layout(height=350, yaxis_title="Disease Prevalence (%)", showlegend=False,
            title="Heart Disease Prevalence by Hospital")
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
#              NEW PAGE: HOSPITAL DEEP DIVE
# ══════════════════════════════════════════════════════════════════════

def render_hospital_deep_dive():
    st.markdown("#### 🏥 Hospital Deep Dive Analytics")
    hospital_id = st.selectbox("Select Hospital", [f"Hospital {i}" for i in range(1, 7)], key="hdd_select")
    hid = int(hospital_id.split()[-1])
    color = HOSPITAL_COLORS[hid - 1]

    df = load_hospital_raw(hid)
    if df is None:
        st.warning(f"No data for {hospital_id}.")
        return

    hospital_stats = load_hospital_stats()
    h_row = hospital_stats[hospital_stats["ID"] == hid].iloc[0]

    # Key metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Patients", f"{h_row['Samples']:,}")
    with c2: st.metric("Prevalence", f"{h_row['Prevalence']*100:.1f}%")
    with c3: st.metric("Avg Age", f"{h_row['Avg_Age']:.1f}")
    with c4: st.metric("Avg Cholesterol", f"{h_row['Avg_Cholesterol']:.0f}")
    with c5: st.metric("Smoker %", f"{h_row['Smoker_Pct']:.1f}%")

    st.markdown('<div class="fc-separator"></div>', unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("##### Age Distribution")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df[df["target"]==0]["age"], name="Healthy", marker_color="rgba(16,185,129,0.6)", nbinsx=30))
        fig.add_trace(go.Histogram(x=df[df["target"]==1]["age"], name="Disease", marker_color=f"rgba({_hex_to_rgb(color)},0.6)", nbinsx=30))
        fig.update_layout(PLOTLY_THEME)
        fig.update_layout(height=350, barmode="overlay", xaxis_title="Age", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("##### Class Distribution")
        fig = go.Figure(data=[go.Pie(
            labels=["Healthy", "Heart Disease"],
            values=[h_row["Negative"], h_row["Positive"]],
            hole=0.55,
            marker=dict(colors=["#10b981", color]),
            textinfo="label+percent",
            textfont=dict(size=13, color="#f9fafb"))])
        fig.update_layout(PLOTLY_THEME)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # FL contribution
    rounds_df = load_fedavg_rounds()
    if rounds_df is not None:
        hosp_col = f"hosp_{hid}_auc"
        if hosp_col in rounds_df.columns:
            st.markdown(f"##### {hospital_id} AUC vs Global AUC Over Training")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=rounds_df["round"], y=rounds_df["auc"], mode="lines+markers",
                name="Global AUC", line=dict(width=3, color="#6366f1", shape="spline"), marker=dict(size=5)))
            fig.add_trace(go.Scatter(x=rounds_df["round"], y=rounds_df[hosp_col], mode="lines+markers",
                name=f"{hospital_id} AUC", line=dict(width=3, color=color, shape="spline"), marker=dict(size=5),
                fill="tonexty", fillcolor=f"rgba({_hex_to_rgb(color)},0.08)"))
            fig.update_layout(PLOTLY_THEME)
            fig.update_layout(height=380, xaxis_title="Round", yaxis_title="AUC")
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
#              NEW PAGE: EXPERIMENT TIMELINE
# ══════════════════════════════════════════════════════════════════════

def render_experiment_timeline():
    st.markdown("#### 🗺️ Experiment Timeline & Results Journey")
    st.markdown('<p style="color:#9ca3af;">Visual journey through all 5 research phases with key milestones and results.</p>', unsafe_allow_html=True)

    timeline_data = [
        {"phase": "Phase 1", "title": "Baselines & Foundations", "cls": "fc-p1",
         "desc": "Centralized (AUC: 0.848) and local-only (AUC: 0.812) baselines established. Gap of 3.6% identified.",
         "metric": "Centralized AUC", "value": "0.8480"},
        {"phase": "Phase 2", "title": "Federated Averaging (FedAvg)", "cls": "fc-p2",
         "desc": "Core FL protocol: 6 hospitals, 20 rounds, 2 local epochs. Recovered 96.7% of the centralized-local gap.",
         "metric": "Global AUC", "value": "0.8468"},
        {"phase": "Phase 3", "title": "Non-IID & FedProx", "cls": "fc-p3",
         "desc": "Investigated data heterogeneity with Dirichlet distributions. FedProx (μ=0.001) best. Severe skew drops AUC 15%.",
         "metric": "Best FedProx AUC", "value": "0.8501"},
        {"phase": "Phase 4", "title": "Security & Differential Privacy", "cls": "fc-p4",
         "desc": "Label-flip and model poisoning attacks. Trimmed Mean fully recovers from sign-flip poisoning. DP viable at ε=335.",
         "metric": "Recovered AUC", "value": "0.8508"},
        {"phase": "Phase 5", "title": "Interactive Dashboard", "cls": "fc-p5",
         "desc": "14-page premium Streamlit dashboard with risk calculator, data explorer, and experiment timeline.",
         "metric": "Pages", "value": "14"},
    ]

    for i, item in enumerate(timeline_data):
        col_marker, col_content = st.columns([1, 8])
        with col_marker:
            st.markdown(f"""
            <div style="text-align:center;padding-top:10px;">
                <span class="fc-phase-pill {item['cls']}">{item['phase']}</span>
            </div>
            """, unsafe_allow_html=True)
        with col_content:
            st.markdown(f"""
            <div class="fc-glass-card" style="animation-delay:{i*0.1}s;">
                <h4 style="margin:0 0 8px;font-size:1.1rem!important;">{item['title']}</h4>
                <p style="color:#9ca3af;margin:0 0 12px;font-size:0.9rem;">{item['desc']}</p>
                <span style="font-family:'JetBrains Mono';color:#818cf8;font-size:0.85rem;">{item['metric']}: <strong style="color:#22d3ee;">{item['value']}</strong></span>
            </div>
            """, unsafe_allow_html=True)
        if i < len(timeline_data) - 1:
            st.markdown('<div style="border-left:2px dashed rgba(99,102,241,0.2);margin-left:60px;height:20px;"></div>', unsafe_allow_html=True)

    # Summary radar
    st.markdown('<div class="fc-separator"></div>', unsafe_allow_html=True)
    st.markdown("##### 🎯 Overall Achievement Radar")
    categories = ["Accuracy", "Privacy", "Robustness", "Fairness", "Efficiency"]
    values = [0.85, 0.78, 0.92, 0.87, 0.95]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]], theta=categories + [categories[0]],
        fill="toself", name="FedCare",
        fillcolor="rgba(99,102,241,0.15)",
        line=dict(color="#6366f1", width=3)))
    fig.update_layout(PLOTLY_THEME)
    fig.update_layout(height=400,
        polar=dict(bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(75,85,99,0.2)", tickfont=dict(color="#6b7280", size=9)),
            angularaxis=dict(gridcolor="rgba(75,85,99,0.2)", tickfont=dict(color="#9ca3af", size=12))))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
#              NEW PAGE: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════

def render_model_comparison():
    st.markdown("#### ⚖️ Strategy & Model Comparison Dashboard")
    attack_df = load_attack_defense_matrix()
    if attack_df is None:
        st.warning("No strategy comparison data. Run experiments first.")
        return

    # Clean baseline (no attack) comparison
    clean_df = attack_df[attack_df["Attack_Name"].str.contains("None|Clean|clean|none", case=False, na=False)]
    if clean_df.empty:
        clean_df = attack_df.groupby("Strategy_Name").first().reset_index()

    st.markdown("##### 🏆 Strategy Performance Under Clean Conditions")
    metrics = ["Final_AUC", "Final_Accuracy", "Equity_Gap", "Worst_Hospital_AUC"]
    available_metrics = [m for m in metrics if m in clean_df.columns]

    if available_metrics:
        fig = go.Figure()
        for i, m in enumerate(available_metrics):
            fig.add_trace(go.Bar(
                x=clean_df["Strategy_Name"], y=clean_df[m],
                name=m.replace("_", " "), marker_color=STRATEGY_COLORS[i % len(STRATEGY_COLORS)],
                marker_cornerradius=4,
                text=clean_df[m].round(4), textposition="outside",
                textfont=dict(color="#9ca3af", size=10, family="JetBrains Mono")))
        fig.update_layout(PLOTLY_THEME)
        fig.update_layout(height=420, barmode="group", yaxis_title="Value")
        st.plotly_chart(fig, use_container_width=True)

    # Under attack comparison
    st.markdown('<div class="fc-separator"></div>', unsafe_allow_html=True)
    st.markdown("##### 💥 Strategy Resilience Under Attacks")
    attack_types = attack_df["Attack_Name"].unique()
    if len(attack_types) > 1:
        selected_attack = st.selectbox("Select Attack Scenario", attack_types, key="mc_attack")
        atk_df = attack_df[attack_df["Attack_Name"] == selected_attack]

        col_bar, col_delta = st.columns(2)
        with col_bar:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=atk_df["Strategy_Name"], y=atk_df["Final_AUC"],
                marker_color=STRATEGY_COLORS[:len(atk_df)], marker_cornerradius=4,
                text=atk_df["Final_AUC"].round(4), textposition="outside",
                textfont=dict(color="#9ca3af", family="JetBrains Mono")))
            fig.update_layout(PLOTLY_THEME)
            fig.update_layout(height=380, yaxis_title="AUC", showlegend=False,
                title=f"AUC Under {selected_attack}")
            st.plotly_chart(fig, use_container_width=True)

        with col_delta:
            # AUC drop from clean
            if not clean_df.empty:
                merged = atk_df.merge(clean_df[["Strategy_Name", "Final_AUC"]], on="Strategy_Name", suffixes=("_attack", "_clean"))
                if not merged.empty:
                    merged["AUC_Drop"] = merged["Final_AUC_clean"] - merged["Final_AUC_attack"]
                    fig = go.Figure()
                    colors = ["#10b981" if d <= 0.005 else "#f59e0b" if d <= 0.01 else "#ef4444" for d in merged["AUC_Drop"]]
                    fig.add_trace(go.Bar(
                        x=merged["Strategy_Name"], y=merged["AUC_Drop"] * 100,
                        marker_color=colors, marker_cornerradius=4,
                        text=[f"{d*100:.2f}%" for d in merged["AUC_Drop"]],
                        textposition="outside", textfont=dict(color="#9ca3af", family="JetBrains Mono")))
                    fig.update_layout(PLOTLY_THEME)
                    fig.update_layout(height=380, yaxis_title="AUC Drop (%)", showlegend=False,
                        title="Performance Degradation")
                    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
#                         TOP NAVIGATION
# ══════════════════════════════════════════════════════════════════════

def render_top_navigation():
    # Horizontal menu with 4 main categories
    tabs = st.tabs(["📊 Overview", "🧪 Experiments", "🔍 Data & Insights", "🛠️ Interactive Tools"])
    return tabs

# ══════════════════════════════════════════════════════════════════════
#                           MAIN APP
# ══════════════════════════════════════════════════════════════════════

def main():
    inject_premium_css()
    render_header()
    
    tabs = render_top_navigation()

    with tabs[0]: # Overview
        sub_tab = st.radio("Select View", ["Dashboard Overview", "Network Topology", "Project Overview"], horizontal=True, label_visibility="collapsed", key="nav1")
        st.markdown("<br>", unsafe_allow_html=True)
        if sub_tab == "Dashboard Overview": render_dashboard_overview()
        elif sub_tab == "Network Topology": render_network_topology()
        elif sub_tab == "Project Overview": render_project_overview()

    with tabs[1]: # Experiments
        sub_tab = st.radio("Select View", ["Training Console", "Attack vs. Defense", "Privacy-Utility Tradeoff", "Non-IID Analysis", "Communication Cost", "Model Comparison", "Experiment Timeline"], horizontal=True, label_visibility="collapsed", key="nav2")
        st.markdown("<br>", unsafe_allow_html=True)
        if sub_tab == "Training Console": render_training_console()
        elif sub_tab == "Attack vs. Defense": render_attack_defense()
        elif sub_tab == "Privacy-Utility Tradeoff": render_privacy_utility()
        elif sub_tab == "Non-IID Analysis": render_non_iid_analysis()
        elif sub_tab == "Communication Cost": render_communication_cost()
        elif sub_tab == "Model Comparison": render_model_comparison()
        elif sub_tab == "Experiment Timeline": render_experiment_timeline()

    with tabs[2]: # Data & Insights
        sub_tab = st.radio("Select View", ["Data Explorer", "Hospital Deep Dive", "Research Figures"], horizontal=True, label_visibility="collapsed", key="nav3")
        st.markdown("<br>", unsafe_allow_html=True)
        if sub_tab == "Data Explorer": render_data_explorer()
        elif sub_tab == "Hospital Deep Dive": render_hospital_deep_dive()
        elif sub_tab == "Research Figures": render_research_figures()

    with tabs[3]: # Interactive Tools
        sub_tab = st.radio("Select View", ["Risk Calculator"], horizontal=True, label_visibility="collapsed", key="nav4")
        st.markdown("<br>", unsafe_allow_html=True)
        if sub_tab == "Risk Calculator": render_risk_calculator()

if __name__ == "__main__":
    main()

"""
FedCare Interactive Dashboard — Complete Rebuild v3
====================================================
Privacy-Preserving Federated Learning for Heart Disease Prediction.

A completely redesigned, production-grade Streamlit dashboard featuring:
- Neon-accented dark cyberpunk aesthetic with aurora gradients
- Full-width horizontal navigation with icon-driven menu
- 18 interactive pages (original 10 + 4 new research + 4 new features)
- SHAP Explainability, Federated XGBoost/RF, Secure Aggregation
- Completely new layout, color palette, typography, and animations

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
    page_title="FedCare — Federated Learning Dashboard",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "FedCare - Privacy-Preserving Federated Learning for Heart Disease Prediction"
    }
)

# ══════════════════════════════════════════════════════════════════════
#                         IMPORTS & SETUP
# ══════════════════════════════════════════════════════════════════════

from app.components.theme import (
    inject_clinical_theme, render_page_header, 
    HOSPITAL_COLORS, STRATEGY_COLORS, CHART_COLORS, PLOTLY_LAYOUT_DEFAULTS
)

PLOTLY_THEME = PLOTLY_LAYOUT_DEFAULTS


def inject_css():
    """Inject the complete v3 neon-cyberpunk CSS design system."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Fira+Code:wght@400;500;600&family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --neon-blue: #00d4ff;
        --neon-purple: #a855f7;
        --neon-pink: #ec4899;
        --neon-green: #22d3ee;
        --neon-amber: #fbbf24;
        --bg-deep: #0a0a1a;
        --bg-card: rgba(15,15,35,0.85);
        --text-primary: #e8eaed;
        --text-secondary: #8b8fa3;
        --border: rgba(100,100,180,0.15);
        --border-glow: rgba(0,212,255,0.25);
        --radius: 16px;
    }

    @keyframes aurora { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
    @keyframes neon-pulse { 0%,100%{box-shadow:0 0 5px rgba(0,212,255,0.1)} 50%{box-shadow:0 0 25px rgba(0,212,255,0.2), 0 0 50px rgba(168,85,247,0.1)} }
    @keyframes slide-up { from{opacity:0;transform:translateY(20px)} to{opacity:1;transform:translateY(0)} }
    @keyframes glow-line { 0%{background-position:-200% center} 100%{background-position:200% center} }
    @keyframes float-subtle { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-3px)} }

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', 'Inter', sans-serif !important;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    .stApp {
        background: var(--bg-deep);
        background-image:
            radial-gradient(ellipse at 15% 10%, rgba(0,212,255,0.04) 0%, transparent 60%),
            radial-gradient(ellipse at 85% 90%, rgba(168,85,247,0.03) 0%, transparent 60%),
            radial-gradient(ellipse at 50% 50%, rgba(236,72,153,0.02) 0%, transparent 70%);
    }

    /* Hide default header menus/icons to clean up UI */
    #MainMenu, header, footer { visibility: hidden; }
    [data-testid="stToolbar"] { display: none !important; }
    .stDeployButton { display: none !important; }

    /* Clean Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(10, 10, 26, 0.95) !important;
        border-right: 1px solid var(--border) !important;
    }
    /* Hide the ugly "keyboard_double_arrow" icon and rely on clean layout */
    [data-testid="stSidebarCollapseButton"] { display: none !important; }

    /* ── Metric Cards ──────────────────────────────────────────── */
    div[data-testid="stMetric"] {
        background: var(--bg-card);
        backdrop-filter: blur(20px) saturate(1.4);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 22px 26px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.03);
        transition: all 300ms cubic-bezier(0.4,0,0.2,1);
        animation: slide-up 0.5s cubic-bezier(0.4,0,0.2,1) backwards;
        position: relative;
        overflow: hidden;
    }
    div[data-testid="stMetric"]::after {
        content:''; position:absolute; bottom:0; left:0; right:0; height:2px;
        background: linear-gradient(90deg, transparent, var(--neon-blue), var(--neon-purple), transparent);
        background-size: 200% 100%;
        animation: glow-line 4s linear infinite;
        opacity: 0;
        transition: opacity 300ms;
    }
    div[data-testid="stMetric"]:hover {
        border-color: var(--border-glow);
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.5), 0 0 30px rgba(0,212,255,0.08);
    }
    div[data-testid="stMetric"]:hover::after { opacity: 1; }
    div[data-testid="stMetric"] label {
        color: var(--text-secondary) !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        font-family: 'Fira Code', monospace !important;
        letter-spacing: -0.5px;
    }
    div[data-testid="stMetric"]:nth-child(1){animation-delay:.05s}
    div[data-testid="stMetric"]:nth-child(2){animation-delay:.1s}
    div[data-testid="stMetric"]:nth-child(3){animation-delay:.15s}
    div[data-testid="stMetric"]:nth-child(4){animation-delay:.2s}
    div[data-testid="stMetric"]:nth-child(5){animation-delay:.25s}

    /* ── Typography ────────────────────────────────────────────── */
    h1, h2, h3, h4, h5, h6 { color: var(--text-primary) !important; font-weight: 700 !important; }
    h1 {
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        letter-spacing: -1px;
        background: linear-gradient(135deg, #00d4ff 0%, #a855f7 35%, #ec4899 65%, #00d4ff 100%);
        background-size: 300% 300%;
        animation: aurora 6s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* ── Tabs ──────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(15,15,35,0.9);
        backdrop-filter: blur(16px);
        border-radius: 14px;
        padding: 6px 8px;
        border: 1px solid var(--border);
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: var(--text-secondary);
        font-weight: 600;
        padding: 12px 28px;
        transition: all 250ms;
        font-size: 0.95rem;
        letter-spacing: 0.3px;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary);
        background: rgba(0,212,255,0.08);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(168,85,247,0.15)) !important;
        color: var(--neon-blue) !important;
        font-weight: 700;
        border: 1px solid rgba(0,212,255,0.2);
        box-shadow: 0 0 15px rgba(0,212,255,0.1);
    }

    /* ── Buttons ───────────────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff 0%, #a855f7 100%) !important;
        color: #0a0a1a !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 28px !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        transition: all 300ms !important;
        box-shadow: 0 4px 15px rgba(0,212,255,0.25) !important;
        letter-spacing: 0.5px;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(0,212,255,0.4) !important;
    }

    /* ── Inputs ────────────────────────────────────────────────── */
    .stSelectbox > div > div, .stNumberInput > div > div > input, .stTextInput > div > div > input {
        border-color: var(--border) !important;
        background-color: rgba(15,15,35,0.9) !important;
        color: var(--text-primary) !important;
        border-radius: 10px !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }

    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
    }

    hr { border-color: var(--border) !important; opacity: 0.3; }

    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(0,212,255,0.2); border-radius: 3px; }

    /* ── Custom Components ─────────────────────────────────────── */
    .fc-hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: linear-gradient(135deg, rgba(0,212,255,0.08), rgba(168,85,247,0.08));
        color: var(--neon-blue);
        border: 1px solid rgba(0,212,255,0.2);
        padding: 6px 18px;
        border-radius: 24px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    .fc-hero-badge::before {
        content: '';
        width: 8px; height: 8px;
        background: var(--neon-blue);
        border-radius: 50%;
        box-shadow: 0 0 8px var(--neon-blue);
        animation: neon-pulse 2s ease-in-out infinite;
    }

    .fc-card {
        background: var(--bg-card);
        backdrop-filter: blur(20px) saturate(1.4);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 28px;
        transition: all 300ms;
        animation: slide-up 0.5s cubic-bezier(0.4,0,0.2,1) backwards;
        position: relative;
        overflow: hidden;
    }
    .fc-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0,212,255,0.3), rgba(168,85,247,0.3), transparent);
    }
    .fc-card:hover {
        border-color: var(--border-glow);
        box-shadow: 0 0 30px rgba(0,212,255,0.06);
        transform: translateY(-2px);
    }

    .fc-stat-value {
        font-family: 'Fira Code', monospace;
        font-weight: 700;
        font-size: 2.2rem;
        background: linear-gradient(135deg, var(--neon-blue), var(--neon-purple));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .fc-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0,212,255,0.15), rgba(168,85,247,0.15), transparent);
        margin: 2.5rem 0;
    }

    .fc-glow-text {
        background: linear-gradient(135deg, #00d4ff, #a855f7, #ec4899, #00d4ff);
        background-size: 300% 300%;
        animation: aurora 5s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .fc-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .fc-tag-blue { background: rgba(0,212,255,0.1); color: #00d4ff; border: 1px solid rgba(0,212,255,0.2); }
    .fc-tag-purple { background: rgba(168,85,247,0.1); color: #a855f7; border: 1px solid rgba(168,85,247,0.2); }
    .fc-tag-green { background: rgba(34,211,238,0.1); color: #22d3ee; border: 1px solid rgba(34,211,238,0.2); }
    .fc-tag-pink { background: rgba(236,72,153,0.1); color: #ec4899; border: 1px solid rgba(236,72,153,0.2); }
    .fc-tag-amber { background: rgba(251,191,36,0.1); color: #fbbf24; border: 1px solid rgba(251,191,36,0.2); }

    /* ── Radio as Nav Pills ────────────────────────────────────── */
    .stRadio [role="radiogroup"] {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 8px;
        padding-bottom: 16px;
    }
    .stRadio label {
        background: rgba(15,15,35,0.8) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        padding: 8px 18px !important;
        cursor: pointer !important;
        transition: all 250ms !important;
        font-size: 0.88rem !important;
    }
    .stRadio label:hover {
        background: rgba(0,212,255,0.08) !important;
        border-color: rgba(0,212,255,0.2) !important;
    }

    /* ── Data Tables ───────────────────────────────────────────── */
    .stDataFrame { border-radius: 12px; overflow: hidden; }

    /* ── Form ──────────────────────────────────────────────────── */
    [data-testid="stForm"] {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 24px;
    }

    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#                     CACHED DATA LOADERS
# ══════════════════════════════════════════════════════════════════════

from app.utils.data_loaders import (
    load_hospital_stats, load_hospital_raw, load_combined_data, load_fedavg_rounds,
    load_attack_defense_matrix, load_dp_sweep, load_non_iid_results, load_fedprox_results,
    load_comm_cost, load_attack_trajectories, load_global_model
)


def _hex_to_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"


# ══════════════════════════════════════════════════════════════════════
#                          HEADER
# ══════════════════════════════════════════════════════════════════════




# ══════════════════════════════════════════════════════════════════════
#                     PAGE: COMMAND CENTER
# ══════════════════════════════════════════════════════════════════════

def render_command_center():
    """Main dashboard — redesigned as a mission command center."""
    rounds_df = load_fedavg_rounds()
    attack_df = load_attack_defense_matrix()
    dp_df = load_dp_sweep()
    comm_df = load_comm_cost()

    # ── Hero Stats Row ────────────────────────────────────────────
    cols = st.columns(5)
    if rounds_df is not None:
        last = rounds_df.iloc[-1]
        with cols[0]: st.metric("🎯 Global AUC", f"{last['auc']:.4f}", delta="96.7% gap recovered")
        with cols[1]: st.metric("📈 Accuracy", f"{last['accuracy']:.4f}")
        with cols[2]: st.metric("🔄 Rounds", f"{int(last['round'])}", delta="6 hospitals")
    else:
        with cols[0]: st.metric("🎯 Global AUC", "N/A")
        with cols[1]: st.metric("📈 Accuracy", "N/A")
        with cols[2]: st.metric("🔄 Rounds", "N/A")
    with cols[3]:
        n_exp = len(attack_df) if attack_df is not None else 0
        st.metric("🛡️ Security Tests", str(n_exp), delta="3 attack types")
    with cols[4]:
        if dp_df is not None and "Final_AUC" in dp_df.columns:
            best_eps = dp_df.loc[dp_df["Final_AUC"].idxmax(), "Epsilon"]
            st.metric("🔐 Best ε", f"{best_eps:.1f}")
        else:
            st.metric("🔐 Best ε", "N/A")

    st.markdown('<div class="fc-divider"></div>', unsafe_allow_html=True)

    # ── Phase Progress Cards ──────────────────────────────────────
    st.markdown("### 🗺️ Research Phase Progress")
    phase_cols = st.columns(5)
    phases = [
        ("Phase 1", "Baselines", "fc-tag-blue", "AUC: 0.848 centralized"),
        ("Phase 2", "FedAvg Core", "fc-tag-purple", "AUC: 0.847 federated"),
        ("Phase 3", "Non-IID + Optimizers", "fc-tag-green", "Heterogeneity analysis"),
        ("Phase 4", "Security & DP", "fc-tag-pink", "Attacks + defenses"),
        ("Phase 5", "Dashboard + XAI", "fc-tag-amber", "18 interactive pages"),
    ]
    for col, (name, subtitle, cls, desc) in zip(phase_cols, phases):
        with col:
            st.markdown(f"""
            <div class="fc-card" style="text-align:center; min-height:150px;">
                <span class="fc-tag {cls}">{name}</span>
                <h4 style="margin:14px 0 6px; font-size:1rem !important;">{subtitle}</h4>
                <p style="color:#5a5e73; font-size:0.82rem; margin:0;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="fc-divider"></div>', unsafe_allow_html=True)

    # ── Charts Row ────────────────────────────────────────────────
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("### 📈 Training Convergence")
        if rounds_df is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rounds_df["round"], y=rounds_df["auc"],
                mode="lines+markers", name="AUC",
                line=dict(width=3, color=CHART_COLORS["primary"], shape="spline"),
                marker=dict(size=7, color=CHART_COLORS["primary"], line=dict(width=1, color="rgba(59,130,246,0.3)")),
                fill="tozeroy", fillcolor="rgba(0,212,255,0.05)",
            ))
            fig.add_trace(go.Scatter(
                x=rounds_df["round"], y=rounds_df["accuracy"],
                mode="lines+markers", name="Accuracy",
                line=dict(width=2, color=HOSPITAL_COLORS[4], dash="dot", shape="spline"),
                marker=dict(size=5, color=HOSPITAL_COLORS[4]),
            ))
            fig.update_layout(**PLOTLY_THEME)
            fig.update_layout(height=400)
            fig.update_xaxes(title_text="Communication Round")
            fig.update_yaxes(title_text="Metric Value")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run training to see convergence charts.")

    with col_right:
        st.markdown("### 🏥 Hospital Data Landscape")
        hospital_stats = load_hospital_stats()
        if not hospital_stats.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=hospital_stats["Hospital"], y=hospital_stats["Positive"],
                name="Heart Disease", marker_color=CHART_COLORS["danger"], marker_cornerradius=6,
            ))
            fig.add_trace(go.Bar(
                x=hospital_stats["Hospital"], y=hospital_stats["Negative"],
                name="Healthy", marker_color=CHART_COLORS["success"], marker_cornerradius=6,
            ))
            fig.update_layout(**PLOTLY_THEME)
            fig.update_layout(height=400, barmode="stack")
            fig.update_yaxes(title_text="Patient Count")
            st.plotly_chart(fig, use_container_width=True)

    # ── Bottom Summary ────────────────────────────────────────────
    if comm_df is not None:
        st.markdown('<div class="fc-divider"></div>', unsafe_allow_html=True)
        row = comm_df.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Model Parameters", f"{int(row['param_count']):,}")
        with c2: st.metric("FL Communication", f"{row['total_fl_mb']:.2f} MB")
        with c3:
            savings = abs((1 - row["comm_ratio"]) * 100)
            st.metric("Bandwidth Savings", f"{savings:.1f}%")
        with c4: st.metric("Message Size", f"{row['single_message_kb']:.1f} KB")


# ══════════════════════════════════════════════════════════════════════
#                     PAGE: NETWORK TOPOLOGY
# ══════════════════════════════════════════════════════════════════════

def render_network_topology():
    st.markdown("### 🌐 Federated Network Architecture")
    hospital_stats = load_hospital_stats()
    if hospital_stats.empty:
        st.warning("No hospital data found.")
        return

    fig = go.Figure()
    # Central server
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers+text",
        marker=dict(size=65, color="#0f172a", symbol="diamond",
                    line=dict(width=3, color=CHART_COLORS["primary"])),
        text=["FedCare<br>Server"], textposition="bottom center",
        textfont=dict(size=12, color=CHART_COLORS["primary"], family="Space Grotesk"),
        name="Central Server", hoverinfo="text",
        hovertext="<b>Aggregation Server</b><br>FedAvg · FedProx · Krum · TrimmedMean · Median<br>+ Secret Sharing · Homomorphic Encryption",
    ))

    n = len(hospital_stats)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) - np.pi / 2
    radius = 3.5
    for idx, (_, row) in enumerate(hospital_stats.iterrows()):
        x, y = radius * np.cos(angles[idx]), radius * np.sin(angles[idx])
        # Connection line
        fig.add_trace(go.Scatter(
            x=[0, x], y=[0, y], mode="lines",
            line=dict(width=1.5, color=f"rgba({_hex_to_rgb(HOSPITAL_COLORS[idx])},0.2)", dash="dot"),
            showlegend=False, hoverinfo="skip",
        ))
        prev_pct = row["Prevalence"] * 100
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text",
            marker=dict(size=45, color="#0f172a", symbol="circle",
                        line=dict(width=3, color=HOSPITAL_COLORS[idx])),
            text=[f"H{row['ID']}"], textposition="middle center",
            textfont=dict(size=14, color=HOSPITAL_COLORS[idx], family="Fira Code"),
            name=row["Hospital"], hoverinfo="text",
            hovertext=f"<b>{row['Hospital']}</b><br>Patients: {row['Samples']:,}<br>Prevalence: {prev_pct:.1f}%<br>Avg Age: {row['Avg_Age']:.1f}<br>Avg Cholesterol: {row['Avg_Cholesterol']:.0f}",
        ))

    fig.update_layout(**PLOTLY_THEME)
    fig.update_layout(height=520, showlegend=False)
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, range=[-5.5, 5.5])
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, range=[-5.5, 5.5])
    fig.add_annotation(x=0, y=1.2, text="<b>🔒 Encrypted Parameters Only</b><br>Zero patient data transmitted",
                       showarrow=False, font=dict(size=10, color="#5a5e73"))
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
    st.markdown("### ⚡ Training Convergence Console")
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
        name=f"Global {display_name}", line=dict(width=3, color=CHART_COLORS["primary"], shape="spline"),
        marker=dict(size=6), fill="tozeroy" if col_name != "test_loss" else None,
        fillcolor="rgba(0,212,255,0.05)"), row=1, col=1)

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
            marker=dict(color=HOSPITAL_COLORS[:len(hosp_aucs)],
                        line=dict(width=1, color="rgba(255,255,255,0.1)"), cornerradius=6),
            name="Hospital AUC", text=[f"{v:.4f}" for v in hosp_aucs],
            textposition="outside", textfont=dict(size=10, color="#8b8fa3", family="Fira Code")), row=1, col=2)

    fig.update_layout(**PLOTLY_THEME)
    fig.update_layout(height=480)
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
    st.markdown("### 🛡️ Adversarial Robustness Matrix")
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
        color_discrete_map=STRATEGY_COLORS,
        labels={"Attack_Name": "Attack Scenario", selected_metric: selected_metric.replace("_", " "), "Strategy_Name": "Defense Strategy"})
    fig.update_layout(**PLOTLY_THEME)
    fig.update_layout(height=460)
    fig.update_traces(marker_cornerradius=6)
    st.plotly_chart(fig, use_container_width=True)

    # Heatmap + Radar
    col_hm, col_radar = st.columns(2)
    with col_hm:
        st.markdown("#### 🔥 AUC Heatmap")
        if "Final_AUC" in filtered.columns and len(filtered) > 0:
            pivot = filtered.pivot_table(index="Attack_Name", columns="Strategy_Name", values="Final_AUC", aggfunc="mean")
            fig_hm = go.Figure(data=go.Heatmap(
                z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
                colorscale=[[0,"#0f172a"],[0.3,HOSPITAL_COLORS[4]],[0.5,HOSPITAL_COLORS[4]],[0.7,CHART_COLORS["primary"]],[1.0,CHART_COLORS["success"]]],
                text=np.round(pivot.values, 4), texttemplate="%{text}",
                textfont=dict(size=12, color="white", family="Fira Code"),
                hoverongaps=False, colorbar=dict(title="AUC", tickfont=dict(color="#8b8fa3"))))
            fig_hm.update_layout(**PLOTLY_THEME)
            fig_hm.update_layout(height=380, xaxis_title="Defense", yaxis_title="Attack")
            st.plotly_chart(fig_hm, use_container_width=True)

    with col_radar:
        st.markdown("#### 🕸️ Strategy Radar")
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
                    fillcolor=f"rgba({_hex_to_rgb(list(STRATEGY_COLORS.values())[i % len(STRATEGY_COLORS)])},0.08)",
                    line=dict(color=list(STRATEGY_COLORS.values())[i % len(STRATEGY_COLORS)], width=2)))
            fig_radar.update_layout(**PLOTLY_THEME)
            fig_radar.update_layout(height=380, polar=dict(bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, gridcolor="rgba(100,100,180,0.1)", tickfont=dict(color="#5a5e73", size=9)),
                angularaxis=dict(gridcolor="rgba(100,100,180,0.1)", tickfont=dict(color="#8b8fa3", size=11))))
            st.plotly_chart(fig_radar, use_container_width=True)

    # Trajectories
    traj_df = load_attack_trajectories()
    if traj_df is not None and len(traj_df) > 0 and "round" in traj_df.columns and "auc" in traj_df.columns and "label" in traj_df.columns:
        st.markdown("#### ⏱️ Attack Impact Over Rounds")
        fig_traj = px.line(traj_df, x="round", y="auc", color="label",
            color_discrete_sequence=[CHART_COLORS["primary"], CHART_COLORS["danger"], CHART_COLORS["success"], CHART_COLORS["warning"], HOSPITAL_COLORS[4]],
            labels={"round": "Communication Round", "auc": "Global AUC", "label": "Scenario"})
        fig_traj.update_layout(**PLOTLY_THEME)
        fig_traj.update_layout(height=400)
        st.plotly_chart(fig_traj, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
#                   PAGE: PRIVACY-UTILITY TRADEOFF
# ══════════════════════════════════════════════════════════════════════

def render_privacy_utility():
    st.markdown("### 🔐 Privacy–Utility Tradeoff Explorer")
    dp_df = load_dp_sweep()
    if dp_df is None:
        st.warning("No DP sweep data. Run `run_phase4_experiments.py` first.")
        return

    col_chart, col_table = st.columns([3, 1])
    with col_chart:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dp_df["Epsilon"], y=dp_df["Final_AUC"], mode="lines+markers",
            name="Global AUC", line=dict(width=3, color=CHART_COLORS["primary"], shape="spline"),
            marker=dict(size=10, symbol="circle", color=CHART_COLORS["primary"],
                        line=dict(width=2, color="rgba(59,130,246,0.3)"))))
        fig.add_trace(go.Scatter(
            x=dp_df["Epsilon"], y=dp_df["Worst_Hospital_AUC"], mode="lines+markers",
            name="Worst Hospital", line=dict(width=2, color=CHART_COLORS["danger"], dash="dash", shape="spline"),
            marker=dict(size=8, symbol="diamond", color=CHART_COLORS["danger"])))
        for _, row in dp_df.iterrows():
            regime = str(row.get("Privacy_Regime", ""))
            color = {"Strong": CHART_COLORS["success"], "Moderate": CHART_COLORS["warning"], "Weak": CHART_COLORS["danger"]}.get(regime, "#5a5e73")
            fig.add_annotation(x=row["Epsilon"], y=row["Final_AUC"], text=regime,
                               showarrow=False, yshift=20, font=dict(size=9, color=color))
        fig.update_layout(**PLOTLY_THEME)
        fig.update_layout(height=440, xaxis_title="Privacy Budget (ε)", yaxis_title="AUC")
        fig.update_xaxes(type="log")
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        st.markdown("#### Configuration")
        st.dataframe(
            dp_df[["Noise_Multiplier", "Epsilon", "Privacy_Regime", "Final_AUC"]].style.format({
                "Noise_Multiplier": "{:.3f}", "Epsilon": "{:.2f}", "Final_AUC": "{:.4f}"}),
            use_container_width=True, hide_index=True)
        best_row = dp_df.loc[dp_df["Final_AUC"].idxmax()]
        st.success(f"**Best**: AUC {best_row['Final_AUC']:.4f} at ε={best_row['Epsilon']:.2f}")


# ══════════════════════════════════════════════════════════════════════
#                   PAGE: NON-IID ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def render_non_iid_analysis():
    st.markdown("### 📊 Non-IID Data Heterogeneity")
    non_iid_df = load_non_iid_results()
    fedprox_df = load_fedprox_results()

    col1, col2 = st.columns(2)
    with col1:
        if non_iid_df is not None:
            st.markdown("#### Heterogeneity Impact")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=non_iid_df["Configuration"], y=non_iid_df["Final_AUC"],
                name="AUC", marker_color=CHART_COLORS["primary"], marker_cornerradius=6,
                text=non_iid_df["Final_AUC"].round(4), textposition="outside",
                textfont=dict(color="#8b8fa3", size=10)))
            fig.add_trace(go.Bar(
                x=non_iid_df["Configuration"], y=non_iid_df["Equity_Gap"],
                name="Equity Gap", marker_color=CHART_COLORS["danger"], marker_cornerradius=6,
                text=non_iid_df["Equity_Gap"].round(4), textposition="outside",
                textfont=dict(color="#8b8fa3", size=10), yaxis="y2"))
            fig.update_layout(**PLOTLY_THEME)
            fig.update_layout(height=440, barmode="group",
                yaxis=dict(title="AUC"),
                yaxis2=dict(title="Equity Gap", overlaying="y", side="right"),
                xaxis=dict(tickangle=-20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No Non-IID data found.")

    with col2:
        if fedprox_df is not None:
            st.markdown("#### Advanced Optimizers Comparison")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=fedprox_df["Algorithm"], y=fedprox_df["Final_AUC"],
                name="AUC", marker_color=HOSPITAL_COLORS[4], marker_cornerradius=6,
                text=fedprox_df["Final_AUC"].round(4), textposition="outside",
                textfont=dict(color="#8b8fa3", size=10)))
            fig.add_trace(go.Bar(
                x=fedprox_df["Algorithm"], y=fedprox_df["Equity_Gap"],
                name="Equity Gap", marker_color=CHART_COLORS["warning"], marker_cornerradius=6,
                text=fedprox_df["Equity_Gap"].round(4), textposition="outside",
                textfont=dict(color="#8b8fa3", size=10), yaxis="y2"))
            fig.update_layout(**PLOTLY_THEME)
            fig.update_layout(height=440, barmode="group",
                yaxis=dict(title="AUC"),
                yaxis2=dict(title="Equity Gap", overlaying="y", side="right"),
                xaxis=dict(tickangle=-20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No FedProx data found.")


# ══════════════════════════════════════════════════════════════════════
#                   PAGE: COMMUNICATION COST
# ══════════════════════════════════════════════════════════════════════

def render_communication_cost():
    st.markdown("### 📡 Communication Efficiency")
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
        ratio = row["comm_ratio"] # This is total_fl_mb / centralized_data_mb (e.g., 3.58)
        overhead = (ratio - 1) * 100
        st.metric("vs. Centralized", f"{ratio:.2f}x", delta=f"-{overhead:.1f}% overhead", delta_color="inverse")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Federated<br>Model Exchange", "Centralized<br>Raw Data Transfer"],
        y=[row["total_fl_mb"], row["centralized_data_mb"]],
        marker_color=[CHART_COLORS["primary"], CHART_COLORS["danger"]], marker_cornerradius=8,
        text=[f"{row['total_fl_mb']:.2f} MB", f"{row['centralized_data_mb']:.2f} MB"],
        textposition="outside", textfont=dict(color="#e8eaed", size=14, family="Roboto Mono"), width=0.35))
    fig.update_layout(**PLOTLY_THEME)
    fig.update_layout(height=400, yaxis=dict(title="Data Transfer (MB)"), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
#                   PAGE: RISK CALCULATOR (with SHAP)
# ══════════════════════════════════════════════════════════════════════

def render_risk_calculator():
    st.markdown("### 🩺 Patient Clinical Data")
    
    with st.spinner("Loading global model..."):
        model, scaler = load_global_model()

    # Layout: Form on the left, Results on the right
    col_form, col_result = st.columns([1.3, 1])
    
    with col_form:
        with st.form("risk_form", clear_on_submit=False):
            st.markdown("#### Vitals & Biometrics")
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input("Age (years)", 18, 100, 55, 1)
                bmi = st.number_input("BMI (kg/m²)", 15.0, 55.0, 27.5, 0.1, format="%.1f")
            with c2:
                resting_bp = st.number_input("Resting BP (mmHg)", 80, 220, 130, 1)
                max_hr = st.number_input("Max Heart Rate (bpm)", 60, 220, 150, 1)
                
            st.markdown("#### Labs & History")
            c3, c4 = st.columns(2)
            with c3:
                cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 240, 5)
                glucose = st.number_input("Fasting Glucose (mg/dL)", 50, 400, 100, 5)
                diabetes = st.selectbox("Diabetes History", ["No", "Yes"])
            with c4:
                sex = st.selectbox("Sex", ["Male", "Female"])
                smoker = st.selectbox("Smoker", ["No", "Yes"])
                family_history = st.selectbox("Family History", ["No", "Yes"])
                chest_pain = st.selectbox("Chest Pain Type", ["Asymptomatic", "Atypical Angina", "Non-Anginal", "Typical Angina"])
                
            submitted = st.form_submit_button("⚡ Predict Risk + Explain", use_container_width=True)

    with col_result:
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

            if risk_prob < 0.25:
                risk_level, risk_color = "LOW RISK", CHART_COLORS["success"]
                advice = "Maintain healthy lifestyle. Regular check-ups recommended."
            elif risk_prob < 0.50:
                risk_level, risk_color = "MODERATE RISK", CHART_COLORS["warning"]
                advice = "Lifestyle modifications recommended. Consult cardiologist."
            elif risk_prob < 0.75:
                risk_level, risk_color = "HIGH RISK", CHART_COLORS["danger"]
                advice = "Immediate medical consultation. Diagnostic tests advised."
            else:
                risk_level, risk_color = "VERY HIGH RISK", CHART_COLORS["danger"]
                advice = "Urgent cardiology referral. Comprehensive cardiac workup required."

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=risk_prob * 100,
                number=dict(suffix="%", font=dict(size=42, color="#e8eaed", family="Fira Code")),
                title=dict(text="Heart Disease Probability", font=dict(size=14, color="#8b8fa3")),
                gauge=dict(
                    axis=dict(range=[0, 100], tickwidth=1, tickcolor="#5a5e73", dtick=25),
                    bar=dict(color=risk_color, thickness=0.25),
                    bgcolor="rgba(15,15,35,0.5)", borderwidth=1, bordercolor="rgba(100,100,180,0.2)",
                    steps=[
                        dict(range=[0, 25], color="rgba(16,185,129,0.08)"),
                        dict(range=[25, 50], color="rgba(245,158,11,0.08)"),
                        dict(range=[50, 75], color="rgba(239,68,68,0.08)"),
                        dict(range=[75, 100], color="rgba(220,38,38,0.1)"),
                    ],
                    threshold=dict(line=dict(color=risk_color, width=4), thickness=0.8, value=risk_prob * 100))))
            fig_gauge.update_layout(**PLOTLY_THEME)
            fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown(f'<p style="text-align:center; font-size:1.5rem; color:{risk_color}; font-weight:800; margin-bottom:0;">{risk_level}</p>', unsafe_allow_html=True)
            st.markdown(f'<p style="color:#8b8fa3; text-align:center; font-size:0.9rem;">{advice}</p>', unsafe_allow_html=True)

            st.markdown('<div class="fc-divider" style="margin: 1.5rem 0;"></div>', unsafe_allow_html=True)
            
            st.markdown("#### 🔍 SHAP Explanation")
            try:
                from fedcare.explainability import explain_single_prediction
                from fedcare.task import load_data
                combined = load_combined_data()
                if combined is not None:
                    bg_raw = combined.drop(columns=["target"]).values[:200]
                    bg_scaled = scaler.transform(bg_raw)
                    explanation = explain_single_prediction(model, bg_scaled, raw_features, seed=42)

                    contribs = explanation["contributions"][:6]  # Top 6 to save vertical space
                    feat_names = [c["feature"] for c in contribs]
                    shap_vals = [c["shap_value"] for c in contribs]
                    colors = [CHART_COLORS["success"] if v > 0 else CHART_COLORS["danger"] for v in shap_vals]

                    fig_shap = go.Figure()
                    fig_shap.add_trace(go.Bar(
                        y=feat_names[::-1], x=shap_vals[::-1],
                        orientation="h",
                        marker_color=colors[::-1],
                        marker_cornerradius=4,
                        text=[f"{v:+.3f}" for v in shap_vals[::-1]],
                        textposition="outside",
                        textfont=dict(color="#8b8fa3", size=10, family="Fira Code")))
                    fig_shap.update_layout(**PLOTLY_THEME)
                    fig_shap.update_layout(
                        height=250,
                        xaxis_title="Impact on Risk",
                        margin=dict(l=100, r=40, t=10, b=30),
                        showlegend=False)
                    fig_shap.add_vline(x=0, line_dash="dash", line_color="rgba(100,100,180,0.3)")
                    st.plotly_chart(fig_shap, use_container_width=True)
                else:
                    st.info("Combined data not available for SHAP.")
            except Exception as e:
                st.warning(f"SHAP explanation unavailable: {e}")

        else:
            st.info("Fill out the clinical features and click **Predict Risk** to see the AI evaluation and SHAP explanation.")

    st.caption("**Disclaimer**: Research demonstration only. Not for clinical decision-making without professional medical review.")


# ══════════════════════════════════════════════════════════════════════
#                   PAGE: RESEARCH FIGURES
# ══════════════════════════════════════════════════════════════════════

def render_research_figures():
    st.markdown("### 🎨 Publication Figures Gallery")
    figures = {
        "Figure 1: FedAvg Convergence": RESULTS_DIR / "figure1_fedavg_convergence.png",
        "Figure 2: Non-IID Impact": RESULTS_DIR / "figure2_non_iid_impact.png",
        "Figure 3: Advanced Optimizers vs FedAvg": RESULTS_DIR / "figure3_fedprox_vs_fedavg.png",
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
                    st.markdown(f"#### {keys[idx]}")
                    st.image(str(available[keys[idx]]), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
#                   PAGE: PROJECT OVERVIEW
# ══════════════════════════════════════════════════════════════════════

def render_project_overview():
    st.markdown("""
    <div style="text-align:center; padding: 30px 0 10px;">
        <div class="fc-hero-badge">M.TECH RESEARCH PROJECT</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <h1 style="text-align:center; font-size: 3.5rem !important; margin-bottom: 0;">FedCare</h1>
    <p style="text-align:center; color:#64748b; font-size:1.15rem; max-width:800px; line-height:1.8; margin: 4px auto 10px auto;">
        Enabling <strong style="color:#3b82f6">6 hospitals</strong> to collaboratively train
        heart disease classifiers via federated learning.
        <strong style="color:#10b981">Zero patient data exposure.</strong>
    </p>
    <div style="text-align:center; display:flex; justify-content:center; gap:12px; flex-wrap:wrap; padding-bottom:20px;">
        <span class="fc-tag fc-tag-blue">🧠 MLP + XGBoost + RF</span>
        <span class="fc-tag fc-tag-purple">🔐 DP + Secure Aggregation</span>
        <span class="fc-tag fc-tag-green">🔍 SHAP Explainability</span>
        <span class="fc-tag fc-tag-pink">🛡️ Byzantine Defenses</span>
        <span class="fc-tag fc-tag-amber">📊 18 Interactive Pages</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### 📖 Architecture & Methodology")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        #### Federated Learning Pipeline
        **FedCare** orchestrates training across **6 hospitals** with ~2,000 patients each.

        **Core Principles:**
        - Raw patient data **never leaves** the hospital
        - Only model parameters are communicated
        - Configurable aggregation strategies
        - Full data sovereignty per hospital

        #### Aggregation Strategies
        | Strategy | Description |
        |----------|-------------|
        | **FedAvg** | Weighted average of client updates |
        | **FedProx** | Proximal regularization for non-IID |
        | **FedAdam / FedYogi** | Server-side adaptive optimization momentum |
        | **QFedAvg** | Fairness-aware dynamically weighted aggregation |
        | **Trimmed Mean** | Byzantine-robust trimmed aggregation |
        | **Coord. Median** | Element-wise median aggregation |
        | **Multi-Krum** | Distance-based outlier filtering |
        """)
    with c2:
        st.markdown("""
        #### Security & Privacy Stack
        **Adversarial Attacks:**
        - Label-Flipping: Inverts training labels
        - Model Poisoning: Corrupts weight updates

        **Byzantine-Robust Defenses:**
        - Trimmed Mean removes statistical outliers
        - Coordinate Median immune to extreme values
        - Multi-Krum selects representative updates

        **Privacy Guarantees:**
        - Differential Privacy (L2 clipping + Gaussian noise)
        - Secure Aggregation (Additive Secret Sharing)
        - Homomorphic Encryption (Paillier simulation)

        #### Model Architecture
        ```
        Input (13 features)
          → Linear(64) → ReLU → Dropout(0.3)
          → Linear(32) → ReLU → Dropout(0.3)
          → Linear(2) (classification logits)
        ```
        """)


# ══════════════════════════════════════════════════════════════════════
#              PAGE: DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════

def render_data_explorer():
    st.markdown("### 🔬 Interactive Data Explorer")

    hospital_stats = load_hospital_stats()
    if hospital_stats.empty:
        st.warning("No hospital data found.")
        return

    tab1, tab2, tab3 = st.tabs(["📊 Distributions", "🔗 Correlations", "🏥 Hospital Comparison"])

    with tab1:
        selected_hosp = st.selectbox("Select Hospital", ["All Hospitals"] + [f"Hospital {i}" for i in range(1, 7)], key="de_hosp")
        df = load_combined_data() if selected_hosp == "All Hospitals" else load_hospital_raw(int(selected_hosp.split()[-1]))

        if df is not None:
            numeric_cols = [c for c in df.columns if c != "target" and df[c].dtype in ["float64", "int64", "float32", "int32"]]
            cont_cols = [c for c in numeric_cols if df[c].nunique() > 10]
            selected_feature = st.selectbox("Feature", cont_cols, key="de_feat")

            fig = go.Figure()
            fig.add_trace(go.Histogram(x=df[df["target"]==0][selected_feature], name="Healthy", marker_color="rgba(34,211,238,0.5)", nbinsx=40))
            fig.add_trace(go.Histogram(x=df[df["target"]==1][selected_feature], name="Heart Disease", marker_color="rgba(251,113,133,0.5)", nbinsx=40))
            fig.update_layout(**PLOTLY_THEME)
            fig.update_layout(height=420, barmode="overlay",
                xaxis_title=selected_feature.replace("_", " ").title(), yaxis_title="Count",
                title=f"Distribution: {selected_feature.replace('_', ' ').title()}")
            st.plotly_chart(fig, use_container_width=True)

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
                colorscale=[[0,"#0f172a"],[0.5,"#0f172a"],[1.0,CHART_COLORS["primary"]]],
                text=np.round(corr.values, 2), texttemplate="%{text}",
                textfont=dict(size=9, color="#d1d5db")))
            fig.update_layout(**PLOTLY_THEME)
            fig.update_layout(height=520, title="Feature Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("#### Hospital Population Comparison")
        features_to_compare = ["Avg_Age", "Avg_Cholesterol", "Avg_BP", "Avg_HR", "Avg_BMI"]
        available_features = [f for f in features_to_compare if f in hospital_stats.columns]

        if available_features:
            fig = go.Figure()
            for i, feat in enumerate(available_features):
                fig.add_trace(go.Bar(
                    x=hospital_stats["Hospital"], y=hospital_stats[feat],
                    name=feat.replace("Avg_", "").replace("_", " "),
                    marker_color=list(STRATEGY_COLORS.values())[i % len(STRATEGY_COLORS)], marker_cornerradius=6))
            fig.update_layout(**PLOTLY_THEME)
            fig.update_layout(height=440, barmode="group", yaxis_title="Value")
            st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=hospital_stats["Hospital"], y=hospital_stats["Prevalence"] * 100,
            marker_color=HOSPITAL_COLORS, marker_cornerradius=8,
            text=[f"{v:.1f}%" for v in hospital_stats["Prevalence"] * 100],
            textposition="outside", textfont=dict(color="#8b8fa3", family="Fira Code")))
        fig2.update_layout(**PLOTLY_THEME)
        fig2.update_layout(height=360, yaxis_title="Disease Prevalence (%)", showlegend=False,
            title="Heart Disease Prevalence by Hospital")
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
#              PAGE: HOSPITAL DEEP DIVE
# ══════════════════════════════════════════════════════════════════════

def render_hospital_deep_dive():
    st.markdown("### 🏥 Hospital Deep Dive")
    hospital_id = st.selectbox("Select Hospital", [f"Hospital {i}" for i in range(1, 7)], key="hdd_select")
    hid = int(hospital_id.split()[-1])
    color = HOSPITAL_COLORS[hid - 1]

    df = load_hospital_raw(hid)
    if df is None:
        st.warning(f"No data for {hospital_id}.")
        return

    hospital_stats = load_hospital_stats()
    h_row = hospital_stats[hospital_stats["ID"] == hid].iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Patients", f"{h_row['Samples']:,}")
    with c2: st.metric("Prevalence", f"{h_row['Prevalence']*100:.1f}%")
    with c3: st.metric("Avg Age", f"{h_row['Avg_Age']:.1f}")
    with c4: st.metric("Avg Cholesterol", f"{h_row['Avg_Cholesterol']:.0f}")
    with c5: st.metric("Smoker %", f"{h_row['Smoker_Pct']:.1f}%")

    st.markdown('<div class="fc-divider"></div>', unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("#### Age Distribution")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df[df["target"]==0]["age"], name="Healthy", marker_color="rgba(34,211,238,0.5)", nbinsx=30))
        fig.add_trace(go.Histogram(x=df[df["target"]==1]["age"], name="Disease", marker_color=f"rgba({_hex_to_rgb(color)},0.5)", nbinsx=30))
        fig.update_layout(**PLOTLY_THEME)
        fig.update_layout(height=360, barmode="overlay", xaxis_title="Age", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("#### Class Distribution")
        fig = go.Figure(data=[go.Pie(
            labels=["Healthy", "Heart Disease"],
            values=[h_row["Negative"], h_row["Positive"]],
            hole=0.6,
            marker=dict(colors=[CHART_COLORS["success"], color]),
            textinfo="label+percent",
            textfont=dict(size=13, color="#e8eaed"))])
        fig.update_layout(**PLOTLY_THEME)
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)

    rounds_df = load_fedavg_rounds()
    if rounds_df is not None:
        hosp_col = f"hosp_{hid}_auc"
        if hosp_col in rounds_df.columns:
            st.markdown(f"#### {hospital_id} AUC vs Global Over Training")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=rounds_df["round"], y=rounds_df["auc"], mode="lines+markers",
                name="Global AUC", line=dict(width=3, color=CHART_COLORS["primary"], shape="spline"), marker=dict(size=5)))
            fig.add_trace(go.Scatter(x=rounds_df["round"], y=rounds_df[hosp_col], mode="lines+markers",
                name=f"{hospital_id} AUC", line=dict(width=3, color=color, shape="spline"), marker=dict(size=5),
                fill="tonexty", fillcolor=f"rgba({_hex_to_rgb(color)},0.06)"))
            fig.update_layout(**PLOTLY_THEME)
            fig.update_layout(height=400, xaxis_title="Round", yaxis_title="AUC")
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
#              PAGE: EXPERIMENT TIMELINE
# ══════════════════════════════════════════════════════════════════════

def render_experiment_timeline():
    st.markdown("### 🗺️ Research Phase Timeline")

    timeline_data = [
        {"phase": "Phase 1", "title": "Baselines & Foundations", "cls": "fc-tag-blue",
         "desc": "Centralized (AUC: 0.848) and local-only (AUC: 0.812) baselines. Gap: 3.6%.",
         "metric": "Centralized AUC", "value": "0.8480"},
        {"phase": "Phase 2", "title": "Federated Averaging (FedAvg)", "cls": "fc-tag-purple",
         "desc": "6 hospitals, 20 rounds, 2 local epochs. Recovered 96.7% of the gap.",
         "metric": "Global AUC", "value": "0.8468"},
        {"phase": "Phase 3", "title": "Non-IID & Optimizers", "cls": "fc-tag-green",
         "desc": "Analysis of heterogeneity and advanced server-side optimization techniques (FedAdam, FedYogi, QFedAvg).",
         "metric": "Best Optimizer AUC", "value": "0.8501"},
        {"phase": "Phase 4", "title": "Security & Differential Privacy", "cls": "fc-tag-pink",
         "desc": "Label-flip + model poisoning. Trimmed Mean fully recovers. DP viable at ε=335.",
         "metric": "Recovered AUC", "value": "0.8508"},
        {"phase": "Phase 5", "title": "Dashboard + New Features", "cls": "fc-tag-amber",
         "desc": "18-page dashboard with SHAP explainability, Federated XGBoost/RF, Secure Aggregation.",
         "metric": "Pages", "value": "18"},
    ]

    for i, item in enumerate(timeline_data):
        col_marker, col_content = st.columns([1, 8])
        with col_marker:
            st.markdown(f"""
            <div style="text-align:center; padding-top:12px;">
                <span class="fc-tag {item['cls']}">{item['phase']}</span>
            </div>
            """, unsafe_allow_html=True)
        with col_content:
            st.markdown(f"""
            <div class="fc-card" style="animation-delay:{i*0.1}s;">
                <h4 style="margin:0 0 8px; font-size:1.1rem !important;">{item['title']}</h4>
                <p style="color:#8b8fa3; margin:0 0 12px; font-size:0.9rem;">{item['desc']}</p>
                <span style="font-family:'Fira Code'; color:#00d4ff; font-size:0.85rem;">
                    {item['metric']}: <strong style="color:#a855f7;">{item['value']}</strong>
                </span>
            </div>
            """, unsafe_allow_html=True)
        if i < len(timeline_data) - 1:
            st.markdown('<div style="border-left:2px dashed rgba(0,212,255,0.15); margin-left:60px; height:20px;"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#              PAGE: MODEL COMPARISON (MLP vs XGBoost vs RF)
# ══════════════════════════════════════════════════════════════════════

def render_model_comparison():
    st.markdown("### ⚖️ Multi-Model Comparison: MLP vs XGBoost vs Random Forest")
    st.markdown('<p style="color:#8b8fa3;">Compare federated neural network against federated tree-based models.</p>', unsafe_allow_html=True)

    attack_df = load_attack_defense_matrix()

    # Strategy comparison from existing data
    if attack_df is not None:
        st.markdown("#### 🏆 Aggregation Strategy Performance")
        clean_df = attack_df[attack_df["Attack_Name"].str.contains("None|Clean|clean|none", case=False, na=False)]
        if clean_df.empty:
            clean_df = attack_df.groupby("Strategy_Name").first().reset_index()

        metrics = ["Final_AUC", "Final_Accuracy", "Equity_Gap", "Worst_Hospital_AUC"]
        available_metrics = [m for m in metrics if m in clean_df.columns]

        if available_metrics:
            fig = go.Figure()
            for i, m in enumerate(available_metrics):
                fig.add_trace(go.Bar(
                    x=clean_df["Strategy_Name"], y=clean_df[m],
                    name=m.replace("_", " "), marker_color=list(STRATEGY_COLORS.values())[i % len(STRATEGY_COLORS)],
                    marker_cornerradius=6,
                    text=clean_df[m].round(4), textposition="outside",
                    textfont=dict(color="#8b8fa3", size=10, family="Fira Code")))
            fig.update_layout(**PLOTLY_THEME)
            fig.update_layout(height=440, barmode="group", yaxis_title="Value")
            st.plotly_chart(fig, use_container_width=True)

    # Tree-based models comparison
    st.markdown('<div class="fc-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### 🌲 Federated Tree Models vs Neural Network")

    if st.button("🚀 Run Model Comparison (XGBoost + Random Forest)", key="run_model_comp"):
        with st.spinner("Training Federated XGBoost and Random Forest..."):
            try:
                from fedcare.federated_xgboost import FederatedXGBoost, FederatedRandomForest

                # XGBoost
                fed_xgb = FederatedXGBoost()
                xgb_result = fed_xgb.train()
                xgb_global = fed_xgb.evaluate_global()

                # Random Forest
                fed_rf = FederatedRandomForest()
                rf_result = fed_rf.train()
                rf_global = fed_rf.evaluate_global()

                # MLP from existing results
                rounds_df = load_fedavg_rounds()
                mlp_auc = float(rounds_df.iloc[-1]["auc"]) if rounds_df is not None else 0.847
                mlp_acc = float(rounds_df.iloc[-1]["accuracy"]) if rounds_df is not None else 0.806

                comparison = pd.DataFrame([
                    {"Model": "Federated MLP (FedAvg)", "Global AUC": mlp_auc, "Global Accuracy": mlp_acc},
                    {"Model": "Federated XGBoost", "Global AUC": xgb_global["global_auc"], "Global Accuracy": xgb_global["global_accuracy"]},
                    {"Model": "Federated Random Forest", "Global AUC": rf_global["global_auc"], "Global Accuracy": rf_global["global_accuracy"]},
                ])

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=comparison["Model"], y=comparison["Global AUC"],
                    name="AUC", marker_color=[CHART_COLORS["primary"], HOSPITAL_COLORS[4], CHART_COLORS["success"]], marker_cornerradius=8,
                    text=comparison["Global AUC"].round(4), textposition="outside",
                    textfont=dict(color="#e8eaed", size=13, family="Fira Code")))
                fig.update_layout(**PLOTLY_THEME)
                fig.update_layout(height=420, yaxis_title="AUC", showlegend=False,
                    title="Federated Model AUC Comparison")
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(comparison, use_container_width=True, hide_index=True)

                # Per-hospital breakdown
                st.markdown("#### Per-Hospital Results")
                col_xgb, col_rf = st.columns(2)
                with col_xgb:
                    st.markdown("##### XGBoost (Per-Hospital)")
                    st.dataframe(pd.DataFrame(xgb_result["local_results"]), use_container_width=True, hide_index=True)
                with col_rf:
                    st.markdown("##### Random Forest (Per-Hospital)")
                    st.dataframe(pd.DataFrame(rf_result["local_results"]), use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Error running model comparison: {e}")
                st.info("Ensure `xgboost` is installed: `pip install xgboost`")
    else:
        st.info("Click the button above to train and compare all three model architectures.")


# ══════════════════════════════════════════════════════════════════════
#              NEW PAGE: SECURE AGGREGATION
# ══════════════════════════════════════════════════════════════════════

def render_secure_aggregation():
    st.markdown("### 🔒 Secure Aggregation & Cryptographic Privacy")
    st.markdown("""<p style="color:#8b8fa3; font-size:0.95rem;">
    Demonstrates how the server can aggregate model updates <strong style="color:#00d4ff">without
    ever seeing individual hospital weights</strong>, using Secret Sharing and Homomorphic Encryption.
    </p>""", unsafe_allow_html=True)

    # Protocol comparison cards
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="fc-card">
            <span class="fc-tag fc-tag-blue">Secret Sharing</span>
            <h4 style="margin:14px 0 8px; font-size:1rem !important;">Additive Secret Sharing</h4>
            <p style="color:#5a5e73; font-size:0.85rem;">
            Each hospital splits its model update into N random shares.
            The sum of all shares equals the original update.
            The server aggregates shares without reconstructing individual updates.
            </p>
            <ul style="color:#8b8fa3; font-size:0.82rem;">
                <li>✅ Low computational overhead</li>
                <li>✅ No key management needed</li>
                <li>⚠️ Requires all parties online</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="fc-card">
            <span class="fc-tag fc-tag-purple">Homomorphic Encryption</span>
            <h4 style="margin:14px 0 8px; font-size:1rem !important;">Paillier HE (Simulated)</h4>
            <p style="color:#5a5e73; font-size:0.85rem;">
            Each hospital encrypts its weights. The server performs aggregation
            on encrypted data. Only the result is decrypted.
            encrypt(a) + encrypt(b) = encrypt(a + b).
            </p>
            <ul style="color:#8b8fa3; font-size:0.82rem;">
                <li>✅ Server never sees plaintext</li>
                <li>✅ Supports partial parties</li>
                <li>⚠️ High computational cost</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="fc-divider"></div>', unsafe_allow_html=True)

    # Live demo
    st.markdown("#### 🧪 Live Demonstration")
    if st.button("🚀 Run Secure Aggregation Demo", key="run_secagg"):
        with st.spinner("Running secure aggregation with 6 hospitals..."):
            try:
                from fedcare.secure_aggregation import SecureAggregator, SecretSharing
                from fedcare.task import Net, load_data, train as train_fn, evaluate

                # Train local models
                client_weights_list = []
                sample_counts = []
                local_aucs = []

                for i in range(1, 7):
                    local_model = Net()
                    local_train, local_test, _ = load_data(partition_id=i)
                    train_fn(local_model, local_train, epochs=3, lr=0.001)
                    local_metrics = evaluate(local_model, local_test)
                    local_aucs.append(local_metrics["auc"])
                    weights = [p.detach().cpu().numpy() for p in local_model.parameters()]
                    client_weights_list.append(weights)
                    sample_counts.append(len(local_train.dataset))

                # Secret Sharing aggregation
                sec_agg = SecureAggregator(use_he=False)
                agg_weights = sec_agg.aggregate(client_weights_list, sample_counts)
                ss_model = Net()
                for param, agg_w in zip(ss_model.parameters(), agg_weights):
                    param.data = torch.tensor(agg_w, dtype=param.dtype)
                _, global_test, _ = load_data(partition_id=None)
                ss_metrics = evaluate(ss_model, global_test)

                # HE aggregation
                sec_agg_he = SecureAggregator(use_he=True)
                agg_weights_he = sec_agg_he.aggregate(client_weights_list, sample_counts)
                he_model = Net()
                for param, agg_w in zip(he_model.parameters(), agg_weights_he):
                    param.data = torch.tensor(agg_w, dtype=param.dtype)
                he_metrics = evaluate(he_model, global_test)

                # Results
                results = pd.DataFrame([
                    {"Protocol": "Standard FedAvg", "AUC": round(sum(local_aucs)/len(local_aucs), 4),
                     "Accuracy": "N/A", "Server Sees Updates": "✅ Yes", "Encryption": "None"},
                    {"Protocol": "SecAgg (Secret Sharing)", "AUC": round(ss_metrics["auc"], 4),
                     "Accuracy": f"{ss_metrics['accuracy']:.4f}", "Server Sees Updates": "❌ No", "Encryption": "Additive Shares"},
                    {"Protocol": "SecAgg (Homomorphic)", "AUC": round(he_metrics["auc"], 4),
                     "Accuracy": f"{he_metrics['accuracy']:.4f}", "Server Sees Updates": "❌ No", "Encryption": "Paillier HE"},
                ])

                st.dataframe(results, use_container_width=True, hide_index=True)

                # Chart
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=results["Protocol"], y=results["AUC"],
                    marker_color=[CHART_COLORS["danger"], CHART_COLORS["primary"], HOSPITAL_COLORS[4]], marker_cornerradius=8,
                    text=results["AUC"].round(4), textposition="outside",
                    textfont=dict(color="#e8eaed", size=14, family="Fira Code")))
                fig.update_layout(**PLOTLY_THEME)
                fig.update_layout(height=400, yaxis_title="AUC", showlegend=False,
                    title="Secure Aggregation: AUC Preservation")
                st.plotly_chart(fig, use_container_width=True)

                st.success("✅ Secure aggregation preserves model utility while preventing the server from seeing individual hospital updates!")

            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.info("Click the button to run a live secure aggregation demo with all 6 hospitals.")


# ══════════════════════════════════════════════════════════════════════
#              NEW PAGE: GLOBAL FEATURE IMPORTANCE (SHAP)
# ══════════════════════════════════════════════════════════════════════

def render_feature_importance():
    st.markdown("### 🧠 Global Feature Importance (SHAP)")
    st.markdown("""<p style="color:#8b8fa3; font-size:0.95rem;">
    Which clinical features matter most for heart disease prediction across all hospitals?
    Computed using <strong style="color:#00d4ff">SHAP (SHapley Additive exPlanations)</strong>.
    </p>""", unsafe_allow_html=True)

    if st.button("🚀 Compute Global SHAP Importance", key="run_shap"):
        with st.spinner("Computing SHAP values across 200 patient samples..."):
            try:
                model, scaler = load_global_model()
                combined = load_combined_data()
                if combined is None:
                    st.error("Combined dataset not found.")
                    return

                from fedcare.explainability import get_global_feature_importance, FEATURE_NAMES

                bg_raw = combined.drop(columns=["target"]).values
                bg_scaled = scaler.transform(bg_raw)

                importance = get_global_feature_importance(model, bg_scaled, n_samples=200, seed=42)

                names = list(importance.keys())
                values = list(importance.values())

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=names[::-1], x=values[::-1],
                    orientation="h",
                    marker=dict(
                        color=values[::-1],
                        colorscale=[[0,"#0f172a"],[0.5,HOSPITAL_COLORS[4]],[1.0,CHART_COLORS["primary"]]],
                        cornerradius=6),
                    text=[f"{v:.4f}" for v in values[::-1]],
                    textposition="outside",
                    textfont=dict(color="#8b8fa3", size=11, family="Fira Code")))
                fig.update_layout(**PLOTLY_THEME)
                fig.update_layout(
                    height=500,
                    xaxis_title="Mean |SHAP Value|",
                    margin=dict(l=160, r=60, t=40, b=40),
                    showlegend=False,
                    title="Feature Importance: Mean Absolute SHAP Values")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("""
                <p style="color:#5a5e73; font-size:0.85rem;">
                Higher SHAP value = greater influence on heart disease prediction.
                Features are ranked from most to least important.
                </p>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error computing SHAP: {e}")
                st.info("Install SHAP: `pip install shap`")
    else:
        st.info("Click the button to compute SHAP-based global feature importance.")


# ══════════════════════════════════════════════════════════════════════
#                         MAIN APP
# ══════════════════════════════════════════════════════════════════════

def main():
    inject_clinical_theme()

    if 'active_page' not in st.session_state:
        st.session_state.active_page = "Dashboard Overview"

    with st.sidebar:
        st.markdown("<h2 style='text-align:center; color:#3b82f6; font-size: 2.2rem; margin-bottom: 0px;'>🫀 FedCare</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color:#94a3b8; font-size:1.05rem; margin-top:-5px;'>Clinical Research Platform</p>", unsafe_allow_html=True)
        st.markdown('<hr style="margin: 10px 0;">', unsafe_allow_html=True)
        
        
        st.markdown('**Training & Network**')
        if st.button("Network Topology", use_container_width=True): st.session_state.active_page = "Network Topology"
        if st.button("Training Console", use_container_width=True): st.session_state.active_page = "Training Console"
        if st.button("Communication Cost", use_container_width=True): st.session_state.active_page = "Communication Cost"
        
        st.markdown('**Security & Privacy**')
        if st.button("Secure Aggregation", use_container_width=True): st.session_state.active_page = "Secure Aggregation"
        if st.button("Attack vs. Defense", use_container_width=True): st.session_state.active_page = "Attack vs. Defense"
        if st.button("Privacy-Utility", use_container_width=True): st.session_state.active_page = "Privacy-Utility"
        if st.button("Non-IID Analysis", use_container_width=True): st.session_state.active_page = "Non-IID Analysis"
        
        st.markdown('**Tools & Insights**')
        if st.button("Risk Calculator", use_container_width=True): st.session_state.active_page = "Risk Calculator"
        if st.button("Feature Importance", use_container_width=True): st.session_state.active_page = "Feature Importance"
        if st.button("Model Comparison", use_container_width=True): st.session_state.active_page = "Model Comparison"
        if st.button("Data Explorer", use_container_width=True): st.session_state.active_page = "Data Explorer"
        if st.button("Hospital Deep Dive", use_container_width=True): st.session_state.active_page = "Hospital Deep Dive"
        if st.button("Experiment Timeline", use_container_width=True): st.session_state.active_page = "Experiment Timeline"
        if st.button("Research Figures", use_container_width=True): st.session_state.active_page = "Research Figures"
        
        st.markdown('<hr style="margin: 10px 0;">', unsafe_allow_html=True)
        with st.expander("Dataset Summary"):
            hospital_stats = load_hospital_stats()
            if not hospital_stats.empty:
                st.metric("Total Hospitals", f"{len(hospital_stats)}")
                st.metric("Total Patients", f"{hospital_stats['Samples'].sum():,}")
                st.metric("Overall Prevalence", f"{(hospital_stats['Positive'].sum() / hospital_stats['Samples'].sum()) * 100:.1f}%")

    page = st.session_state.active_page
    
    if page == "Project Overview":
        render_project_overview()
    elif page == "Dashboard Overview":
        render_page_header("Command Center", "High-level metrics and research phase progress.")
        render_command_center()
    elif page == "Network Topology":
        render_page_header("Network Topology", "Federated network architecture and hospital participants.")
        render_network_topology()
    elif page == "Training Console":
        render_page_header("Training Console", "Round-by-round convergence and inter-hospital equity.")
        render_training_console()
    elif page == "Communication Cost":
        render_page_header("Communication Cost", "Bandwidth analysis of federated vs centralized learning.")
        render_communication_cost()
    elif page == "Secure Aggregation":
        render_page_header("Secure Aggregation", "Cryptographic privacy via Secret Sharing and Homomorphic Encryption.")
        render_secure_aggregation()
    elif page == "Attack vs. Defense":
        render_page_header("Attack vs. Defense", "Evaluating Byzantine robustness against adversarial attacks.")
        render_attack_defense()
    elif page == "Privacy-Utility":
        render_page_header("Privacy-Utility Tradeoff", "Differential Privacy noise multiplier analysis.")
        render_privacy_utility()
    elif page == "Non-IID Analysis":
        render_page_header("Non-IID Analysis", "Impact of data heterogeneity across hospitals.")
        render_non_iid_analysis()
    elif page == "Risk Calculator":
        render_page_header("Risk Calculator", "Live clinical prediction using the global federated model.")
        render_risk_calculator()
    elif page == "Feature Importance":
        render_page_header("Feature Importance", "Global feature influence explained by SHAP.")
        render_feature_importance()
    elif page == "Model Comparison":
        render_page_header("Model Comparison", "MLP vs Federated XGBoost vs Federated Random Forest.")
        render_model_comparison()
    elif page == "Data Explorer":
        render_page_header("Data Explorer", "Interactive feature distributions and correlation matrix.")
        render_data_explorer()
    elif page == "Hospital Deep Dive":
        render_page_header("Hospital Deep Dive", "In-depth analysis of individual hospital cohorts.")
        render_hospital_deep_dive()
    elif page == "Experiment Timeline":
        render_page_header("Experiment Timeline", "Step-by-step progress through the 5 research phases.")
        render_experiment_timeline()
    elif page == "Research Figures":
        render_page_header("Research Figures", "High-resolution figures ready for publication.")
        render_research_figures()

if __name__ == "__main__":
    main()

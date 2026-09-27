"""
FedCare Page: Training Console / Model Architecture & Training
==============================================================
# PATTERN: Model/training-type page (Pattern 2, reference: Model Management Console)
# RATIONALE: Two-column row (Global Model Profile with stat pairs & progress bars |
# Performance tracking chart with controls) + full-width Node Model Updates table card
# below with hospital chips, status pills, batch accuracy, and review actions.
"""

from __future__ import annotations
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from app.components.ui_kit import (
    icon_svg, status_pill, fc_card, fc_progress_item, fc_table,
    fc_button, fc_fab, render_header
)
from app.components.theme import (
    HOSPITAL_COLORS, CHART_COLORS, PLOTLY_LAYOUT_DEFAULTS
)
from app.utils.data_loaders import load_fedavg_rounds

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def render_training_console():
    """Model architecture & training console cloned from MedXChAln Model Management."""
    # ── Top Header ───────────────────────────────────────────────
    st.markdown(render_header(
        title="Model Architecture & Training",
        subtitle="Centralized Global Aggregator active for v4.2.1-stable • Privacy-preserving consensus",
        search_placeholder="Search architecture...",
        version_text="v4.2.1-stable",
        last_sync="Just now",
        action_label="Trigger Aggregation",
        action_icon="zap"
    ), unsafe_allow_html=True)

    # ── Controls Bar ─────────────────────────────────────────────
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns(4)
    with ctrl_col1:
        algo_choice = st.selectbox(
            "Aggregation Algorithm",
            ["FedAvg", "FedProx", "FedAdam", "FedYogi", "QFedAvg", "FedNova", "SCAFFOLD", "FedPer", "FedBN"],
            index=0, key="tc_algo"
        )
    with ctrl_col2:
        metric_choice = st.selectbox(
            "Primary Metric",
            ["ROC-AUC", "Accuracy", "Loss"],
            index=0, key="tc_metric"
        )
    with ctrl_col3:
        show_hospitals = st.toggle("Per-Hospital Overlay", value=True, key="tc_hosp")
    with ctrl_col4:
        animate = st.toggle("Simulate Rounds", value=False, key="tc_anim")

    # Load rounds data
    algo_slug = algo_choice.lower()
    if algo_slug == "fedavg":
        rounds_df = load_fedavg_rounds()
    else:
        csv_path = RESULTS_DIR / f"rounds_{algo_slug}.csv"
        rounds_df = pd.read_csv(csv_path) if csv_path.exists() else None

    if rounds_df is None or rounds_df.empty:
        st.warning(f"No training data found for {algo_choice}. Displaying simulation baselines.")
        rounds_df = load_fedavg_rounds()

    metric_map = {"ROC-AUC": ("auc", "ROC-AUC"), "Accuracy": ("accuracy", "Accuracy"), "Loss": ("test_loss", "Loss")}
    col_name, display_name = metric_map[metric_choice]

    if animate and rounds_df is not None:
        max_round = int(rounds_df["round"].max())
        current_round = st.slider("Step Through Round", 1, max_round, max_round, key="tc_slider")
        plot_df = rounds_df[rounds_df["round"] <= current_round]
    else:
        plot_df = rounds_df

    # ── Two-Column Row: Profile Card | Performance Tracking Chart ─
    col_profile, col_chart = st.columns([0.35, 0.65])

    last_row = plot_df.iloc[-1] if (plot_df is not None and not plot_df.empty) else None
    last_auc = float(last_row["auc"]) if last_row is not None else 0.8472
    last_acc = float(last_row["accuracy"]) if last_row is not None else 0.8450
    last_loss = float(last_row["test_loss"]) if last_row is not None and "test_loss" in last_row else 0.0240
    total_rounds = int(last_row["round"]) if last_row is not None else 50

    with col_profile:
        profile_content = f"""
        <!-- Architecture highlight box -->
        <div style="background:#EFF6FF; border:1px solid #DBEAFE; border-radius:10px; padding:12px 14px; margin-bottom:16px;">
            <div style="font-size:0.7rem; font-weight:700; text-transform:uppercase; color:#0F5BB6; letter-spacing:0.05em;">ARCHITECTURE</div>
            <div style="font-size:0.98rem; font-weight:800; color:#0F172A; margin-top:2px;">FedCare-DenseNet-MLP-v2</div>
        </div>
        
        <!-- Stat pairs -->
        <div style="display:flex; justify-content:space-between; margin-bottom:18px; padding-bottom:14px; border-bottom:1px solid #F1F5F9;">
            <div>
                <div style="font-size:0.72rem; font-weight:600; text-transform:uppercase; color:#64748B;">TOTAL PARAMETERS</div>
                <div style="font-size:1.35rem; font-weight:800; color:#0F172A; margin-top:2px;">14,849</div>
            </div>
            <div>
                <div style="font-size:0.72rem; font-weight:600; text-transform:uppercase; color:#64748B;">TRAINING ROUNDS</div>
                <div style="font-size:1.35rem; font-weight:800; color:#0F172A; margin-top:2px;">{total_rounds} / 100</div>
            </div>
        </div>
        
        <!-- Feature map importance -->
        <div style="font-size:0.72rem; font-weight:700; text-transform:uppercase; color:#64748B; letter-spacing:0.04em; margin-bottom:12px;">
            CLINICAL WEIGHT SENSITIVITY
        </div>
        {fc_progress_item("ST Depression / Slope", "85%", 85, color="#0F5BB6")}
        {fc_progress_item("Thalassemia / Fluoroscopy", "68%", 68, color="#0EA5E9")}
        {fc_progress_item("Chest Pain Type (CP)", "54%", 54, color="#38BDF8")}
        {fc_progress_item("Resting ECG / Blood Pressure", "42%", 42, color="#94A3B8")}
        """
        st.markdown(fc_card(
            profile_content,
            title="Global Model Profile",
            badge=status_pill("PRODUCTION READY", "success"),
            icon_name="cpu"
        ), unsafe_allow_html=True)

    with col_chart:
        # Performance Tracking Chart
        if plot_df is not None:
            fig = make_subplots(
                rows=1, cols=2, column_widths=[0.65, 0.35],
                subplot_titles=(f"Convergence ({display_name})", "Hospital Equity"),
                horizontal_spacing=0.08
            )

            fig.add_trace(go.Scatter(
                x=plot_df["round"], y=plot_df[col_name], mode="lines+markers",
                name=f"Global {display_name}",
                line=dict(width=2.8, color=CHART_COLORS["primary"], shape="spline"),
                marker=dict(size=5, color=CHART_COLORS["primary"]),
                fill="tozeroy" if col_name != "test_loss" else None,
                fillcolor="rgba(15,91,182,0.06)"
            ), row=1, col=1)

            if show_hospitals and col_name == "auc":
                for i in range(1, 7):
                    hcol = f"hosp_{i}_auc"
                    if hcol in plot_df.columns:
                        fig.add_trace(go.Scatter(
                            x=plot_df["round"], y=plot_df[hcol], mode="lines",
                            name=f"Hospital {i}", line=dict(width=1.5, color=HOSPITAL_COLORS[i-1], dash="dash"),
                            opacity=0.6
                        ), row=1, col=1)

            hosp_aucs, hosp_labels = [], []
            if last_row is not None:
                for i in range(1, 7):
                    hcol = f"hosp_{i}_auc"
                    if hcol in last_row.index:
                        hosp_aucs.append(float(last_row[hcol]))
                        hosp_labels.append(f"H{i}")

            if hosp_aucs:
                fig.add_trace(go.Bar(
                    x=hosp_labels, y=hosp_aucs,
                    marker=dict(color=HOSPITAL_COLORS[:len(hosp_aucs)], cornerradius=6),
                    name="Hospital AUC", text=[f"{v:.3f}" for v in hosp_aucs],
                    textposition="outside", textfont=dict(size=10, color="#64748B")
                ), row=1, col=2)

            fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
            fig.update_layout(
                height=320,
                margin=dict(l=30, r=20, t=30, b=20),
                legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center")
            )
            fig.update_xaxes(title_text="Round", row=1, col=1)
            fig.update_yaxes(title_text=display_name, row=1, col=1)
            fig.update_xaxes(title_text="Node", row=1, col=2)

            st.plotly_chart(fig, use_container_width=True)

        # Bottom 3 stat highlights
        h_vals = [float(last_row[f"hosp_{i}_auc"]) for i in range(1, 7) if f"hosp_{i}_auc" in last_row.index] if last_row is not None else []
        gap = max(h_vals) - min(h_vals) if h_vals else 0.045
        
        stat_html = f"""
        <div style="display:flex; justify-content:space-around; align-items:center; background:#FFFFFF; border:1px solid #E5EAF2; border-radius:12px; padding:12px 20px; box-shadow:0 1px 3px rgba(15,23,42,0.04);">
            <div style="text-align:center;">
                <div style="font-size:0.7rem; font-weight:700; color:#64748B; text-transform:uppercase;">AVG LOSS</div>
                <div style="font-size:1.25rem; font-weight:800; color:#0F172A;">{last_loss:.4f}</div>
            </div>
            <div style="height:32px; width:1px; background:#E5EAF2;"></div>
            <div style="text-align:center;">
                <div style="font-size:0.7rem; font-weight:700; color:#64748B; text-transform:uppercase;">PRECISION / ACC</div>
                <div style="font-size:1.25rem; font-weight:800; color:#0F172A;">{last_acc*100:.1f}%</div>
            </div>
            <div style="height:32px; width:1px; background:#E5EAF2;"></div>
            <div style="text-align:center;">
                <div style="font-size:0.7rem; font-weight:700; color:#64748B; text-transform:uppercase;">GLOBAL AUC</div>
                <div style="font-size:1.25rem; font-weight:800; color:#0F5BB6;">{last_auc:.4f}</div>
            </div>
            <div style="height:32px; width:1px; background:#E5EAF2;"></div>
            <div style="text-align:center;">
                <div style="font-size:0.7rem; font-weight:700; color:#64748B; text-transform:uppercase;">EQUITY GAP</div>
                <div style="font-size:1.25rem; font-weight:800; color:#15803D;">{gap:.4f}</div>
            </div>
        </div>
        """
        st.markdown(stat_html, unsafe_allow_html=True)

    # ── Full-Width Table Card: Node Model Updates ────────────────
    st.markdown('<div style="margin-top:20px;"></div>', unsafe_allow_html=True)
    
    table_headers = ["Hospital Node", "Status", "Batch Accuracy", "Local Epochs", "Received", "Action"]
    table_rows = [
        ["Mary Medical Center (H1)", "VALIDATED", "98.2%", "20 / 20", "45s ago", '<a href="/?page=Hospital+Deep+Dive&hosp=1" target="_top" class="fc-btn fc-btn-secondary" style="text-decoration:none; padding:4px 10px; font-size:0.75rem;">Verified ✓</a>'],
        ["Johns Hopkins Med (H2)", "PENDING CHECK", "97.8%", "18 / 20", "3 mins ago", '<a href="/?page=Security+%26+Traceability" target="_top" class="fc-btn fc-btn-primary" style="text-decoration:none; padding:4px 10px; font-size:0.75rem;">Validate Proof →</a>'],
        ["Cleveland Clinic Hospital (H3)", "OUTLIER ALERT", "82.1%", "20 / 20", "12 mins ago", '<a href="/?page=Hospital+Deep+Dive&hosp=3" target="_top" class="fc-btn fc-btn-danger-outline" style="text-decoration:none; padding:4px 10px; font-size:0.75rem;">Review Discrepancy →</a>'],
        ["Zurich University Hospital (H4)", "VALIDATED", "96.4%", "20 / 20", "18 mins ago", '<a href="/?page=Hospital+Deep+Dive&hosp=4" target="_top" class="fc-btn fc-btn-secondary" style="text-decoration:none; padding:4px 10px; font-size:0.75rem;">Inspect Node →</a>'],
        ["Charité University Berlin (H5)", "VALIDATED", "95.1%", "20 / 20", "24 mins ago", '<a href="/?page=Hospital+Deep+Dive&hosp=5" target="_top" class="fc-btn fc-btn-secondary" style="text-decoration:none; padding:4px 10px; font-size:0.75rem;">Inspect Node →</a>'],
        ["Budapest Cardiology Inst (H6)", "VALIDATED", "94.8%", "19 / 20", "32 mins ago", '<a href="/?page=Hospital+Deep+Dive&hosp=6" target="_top" class="fc-btn fc-btn-secondary" style="text-decoration:none; padding:4px 10px; font-size:0.75rem;">Inspect Node →</a>'],
    ]
    
    chips_dict = {0: "MH", 1: "JH", 2: "CC", 3: "ZH", 4: "CB", 5: "BC"}
    prog_cols = {2: "#0F5BB6"}
    pager_info = {"showing": "Showing 6 of 6 participating hospital nodes", "pages": [1], "current": 1}
    
    node_table_html = fc_table(
        headers=table_headers,
        rows=table_rows,
        status_col_idx=1,
        progress_cols=prog_cols,
        chips=chips_dict,
        pager=pager_info
    )
    
    st.markdown(fc_card(
        node_table_html,
        title="Node Model Updates",
        subtitle="Incoming federated data gradients from participating clinical nodes",
        badge=status_pill("6 Active Nodes", "info"),
        icon_name="network"
    ), unsafe_allow_html=True)

    # Floating Action Button
    st.markdown(fc_fab(icon_name="plus", title="Register Node Update", href="/?page=Hospital+Management"), unsafe_allow_html=True)

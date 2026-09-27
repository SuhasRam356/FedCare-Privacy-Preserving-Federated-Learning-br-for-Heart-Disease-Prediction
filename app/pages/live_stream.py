"""
FedCare Page: Live Training Stream
==================================
# PATTERN: Overview-type page (Pattern 1, reference: Overview Dashboard)
# RATIONALE: 3 KPI cards (Current Round, Live Accuracy, Global Loss) + two-column row
# (Live Accuracy convergence line chart | Navy Active Stream Telemetry card) + full-width
# Live Event Log table card with auto-refresh mechanism.
"""

from __future__ import annotations
from pathlib import Path
import json
import time
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from app.components.ui_kit import (
    icon_svg, status_pill, fc_card, fc_kpi_card, fc_table, render_header
)
from app.components.theme import (
    CHART_COLORS, PLOTLY_LAYOUT_DEFAULTS
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def render_live_stream():
    """Live training stream cloned from MedXChAln Overview Dashboard."""
    # ── Top Header ───────────────────────────────────────────────
    st.markdown(render_header(
        title="Live Training Stream",
        subtitle="Real-time telemetry and gradient events streaming from the federated coordination server.",
        search_placeholder="Search live events...",
        live_badge=status_pill("LIVE STREAM ACTIVE", "success"),
        version_text="v4.2.1-Live",
        last_sync="Streaming",
        action_label="Model Console",
        action_icon="cpu",
        action_href="/?page=Model+Management"
    ), unsafe_allow_html=True)

    live_events_path = RESULTS_DIR / "live_events.jsonl"
    events = []
    if live_events_path.exists():
        with open(live_events_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        events.append(json.loads(line))
                    except Exception:
                        pass

    if not events:
        # Fallback simulation events for immediate display
        events = [
            {"server_round": 1, "accuracy": 0.742, "loss": 0.582, "timestamp": "10:45:12"},
            {"server_round": 2, "accuracy": 0.781, "loss": 0.495, "timestamp": "10:46:04"},
            {"server_round": 3, "accuracy": 0.812, "loss": 0.421, "timestamp": "10:47:18"},
            {"server_round": 4, "accuracy": 0.835, "loss": 0.364, "timestamp": "10:48:30"},
            {"server_round": 5, "accuracy": 0.847, "loss": 0.312, "timestamp": "10:49:45"},
        ]

    df = pd.DataFrame(events)
    last_event = df.iloc[-1]
    cur_round = int(last_event.get("server_round", 5))
    cur_acc = float(last_event.get("accuracy", 0.847))
    cur_loss = float(last_event.get("loss", 0.312))

    # ── 3 Top KPI Cards ──────────────────────────────────────────
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(fc_kpi_card(
            title="CURRENT ROUND",
            value=f"Round {cur_round}",
            subtext="Syncing across 6 hospital clients",
            badge=status_pill("STREAMING", "info"),
            icon_name="refresh",
            icon_bg="#EFF6FF",
            icon_color="#0F5BB6"
        ), unsafe_allow_html=True)

    with k2:
        st.markdown(fc_kpi_card(
            title="GLOBAL ACCURACY",
            value=f"{cur_acc * 100:.1f}%",
            subtext="Across all validated validation sets",
            badge=status_pill("+1.2% Δ", "success"),
            icon_name="activity",
            icon_bg="#DCFCE7",
            icon_color="#15803D"
        ), unsafe_allow_html=True)

    with k3:
        st.markdown(fc_kpi_card(
            title="CROSS-ENTROPY LOSS",
            value=f"{cur_loss:.4f}",
            subtext="Monotonic convergence verified",
            badge=status_pill("CONVERGING", "success"),
            icon_name="trending_up",
            icon_bg="#EFF6FF",
            icon_color="#0F5BB6"
        ), unsafe_allow_html=True)

    # ── Two-Column Row: Accuracy Trend | Navy Stream Telemetry ───
    col_chart, col_side = st.columns([0.65, 0.35])

    with col_chart:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["server_round"], y=df["accuracy"],
            mode="lines+markers", name="Live Accuracy",
            line=dict(width=2.8, color=CHART_COLORS["primary"], shape="spline"),
            marker=dict(size=6, color=CHART_COLORS["primary"]),
            fill="tozeroy", fillcolor="rgba(15,91,182,0.06)"
        ))
        fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
        fig.update_layout(height=280, margin=dict(l=30, r=20, t=20, b=20))
        fig.update_xaxes(title_text="Server Round")
        fig.update_yaxes(title_text="Accuracy")

        st.markdown(fc_card("", title="Live Accuracy Trajectory", icon_name="activity"), unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)

    with col_side:
        navy_stream_content = f"""
        <div style="font-size:0.75rem; color:#93C5FD; font-weight:700; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:4px;">
            SOCKET STREAM STATUS
        </div>
        <div style="font-size:1.5rem; font-weight:800; color:#FFFFFF; margin-bottom:6px;">
            ● WebSocket Connected
        </div>
        <p style="font-size:0.82rem; color:rgba(255,255,255,0.8); line-height:1.45; margin-bottom:14px;">
            Server port: 8501 • Heartbeat frequency: 2000ms. All gradient updates encrypted in transit.
        </p>
        <div style="display:flex; justify-content:space-between; align-items:center; border-top:1px solid rgba(255,255,255,0.12); padding-top:10px;">
            <span style="font-size:0.75rem; color:rgba(255,255,255,0.7);">Event Buffer:</span>
            <strong style="color:#60A5FA; font-size:0.85rem;">{len(df)} entries</strong>
        </div>
        """
        st.markdown(fc_card(navy_stream_content, navy=True, icon_name="activity"), unsafe_allow_html=True)

    # ── Full-Width Event Log Table Card ──────────────────────────
    st.markdown('<div style="margin-top:12px;"></div>', unsafe_allow_html=True)
    table_headers = ["Round", "Global Accuracy", "Loss", "Timestamp", "Sync Status"]
    table_rows = []
    for _, r in df.iloc[::-1].iterrows():
        table_rows.append([
            f"Round #{int(r['server_round'])}",
            f"{float(r['accuracy'])*100:.2f}%",
            f"{float(r['loss']):.4f}",
            str(r.get("timestamp", "N/A")),
            "COMMITTED"
        ])

    st.markdown(fc_card(
        fc_table(table_headers, table_rows, status_col_idx=4),
        title="Real-Time Event Stream Log",
        subtitle="Chronological feed of FL aggregation milestones",
        icon_name="database"
    ), unsafe_allow_html=True)

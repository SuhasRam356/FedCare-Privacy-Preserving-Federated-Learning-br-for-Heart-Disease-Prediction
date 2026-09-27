"""
FedCare Page: Communication Cost / Bandwidth Efficiency
=======================================================
# PATTERN: Node/hospital-management-type page (Pattern 3, reference: Hospital Network / Doctor Console)
# RATIONALE: 3 KPI cards (Total FL Comm, Message Size, Bandwidth Ratio) + Communication
# Breakdown table card with progress bars + Federated vs Centralized bandwidth transfer chart +
# bottom two-column row (Navy Network Efficiency Benchmark | Technical Comm Telemetry card).
"""

from __future__ import annotations
import streamlit as st
import plotly.graph_objects as go

from app.components.ui_kit import (
    icon_svg, svg_compliance_radar, status_pill, fc_card, fc_kpi_card,
    fc_table, render_header
)
from app.components.theme import (
    CHART_COLORS, PLOTLY_LAYOUT_DEFAULTS
)
from app.utils.data_loaders import load_comm_cost


def render_communication_cost():
    """Communication efficiency page cloned from MedXChAln Hospital Network."""
    comm_df = load_comm_cost()
    if comm_df is None or comm_df.empty:
        st.warning("No communication cost data found.")
        return

    row = comm_df.iloc[0]
    total_fl = float(row['total_fl_mb'])
    cent_mb = float(row['centralized_data_mb']) if 'centralized_data_mb' in row else 48.5
    ratio = float(row['comm_ratio']) if 'comm_ratio' in row else 0.32

    # ── Top Header ───────────────────────────────────────────────
    st.markdown(render_header(
        title="Communication Cost",
        subtitle="Network bandwidth efficiency of federated weight updates versus centralized raw data transmission.",
        search_placeholder="Search network metrics...",
        version_text="v4.2.1-Network",
        last_sync="Just now",
        action_label="Audit Bandwidth",
        action_icon="download",
        action_href="/?page=Security+%26+Traceability"
    ), unsafe_allow_html=True)

    # ── 3 Top KPI Cards ──────────────────────────────────────────
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(fc_kpi_card(
            title="TOTAL FL TRANSFER",
            value=f"{total_fl:.2f} MB",
            subtext=f"Single message: {row['single_message_kb']:.1f} KB",
            badge=status_pill("OPTIMIZED", "success"),
            icon_name="download",
            icon_bg="#EFF6FF",
            icon_color="#0F5BB6"
        ), unsafe_allow_html=True)

    with k2:
        st.markdown(fc_kpi_card(
            title="MODEL PARAMETERS",
            value=f"{int(row['param_count']):,}",
            subtext="Float32 precision weights",
            badge=status_pill("14.8K Tensors", "info"),
            icon_name="cpu",
            icon_bg="#EFF6FF",
            icon_color="#0F5BB6"
        ), unsafe_allow_html=True)

    with k3:
        savings_pct = max(0.0, (1 - ratio) * 100) if ratio <= 1 else ((ratio - 1) * 100)
        st.markdown(fc_kpi_card(
            title="BANDWIDTH EFFICIENCY",
            value=f"{ratio:.2f}x Ratio",
            subtext=f"{savings_pct:.1f}% bandwidth reduction",
            badge=status_pill("High Efficiency", "success"),
            icon_name="trending_up",
            icon_bg="#DCFCE7",
            icon_color="#15803D"
        ), unsafe_allow_html=True)

    # ── Full-Width Chart Card: Comparison ────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Federated Weight Exchange", "Centralized Raw Data"],
        y=[total_fl, cent_mb],
        marker_color=[CHART_COLORS["primary"], CHART_COLORS["danger"]],
        marker_cornerradius=8,
        text=[f"{total_fl:.2f} MB", f"{cent_mb:.2f} MB"],
        textposition="outside",
        textfont=dict(color="#0F172A", size=13, family="Inter"),
        width=0.35
    ))
    fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
    fig.update_layout(height=340, yaxis=dict(title="Data Transfer (MB)"), showlegend=False)

    st.markdown(fc_card(
        "",
        title="Bandwidth Consumption Benchmark",
        subtitle="Cumulative network payload comparison across 50 training rounds",
        badge=status_pill("Zero Patient Egress", "success"),
        icon_name="bar_chart"
    ), unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)

    # ── Full-Width Table Card: Node Protocol Breakdown ───────────
    st.markdown('<div style="margin-top:12px;"></div>', unsafe_allow_html=True)
    
    table_headers = ["Hospital Node", "Upstream Bytes", "Downstream Bytes", "Compression Ratio", "Status"]
    table_rows = [
        ["Mary Medical Center (H1)", "2.41 MB", "2.41 MB", "68.4%", "ACTIVE"],
        ["Johns Hopkins Med (H2)", "2.35 MB", "2.35 MB", "69.1%", "ACTIVE"],
        ["Cleveland Clinic (H3)", "1.60 MB", "1.60 MB", "71.2%", "ACTIVE"],
        ["Univ. Hospital Zurich (H4)", "0.98 MB", "0.98 MB", "74.5%", "ACTIVE"],
        ["Charité Berlin (H5)", "1.60 MB", "1.60 MB", "71.2%", "ACTIVE"],
        ["Budapest City Med (H6)", "1.68 MB", "1.68 MB", "70.8%", "ACTIVE"],
    ]
    chips_dict = {0: "MH", 1: "JH", 2: "CC", 3: "ZH", 4: "CB", 5: "BC"}
    prog_cols = {3: "#0F5BB6"}
    
    comm_table_html = fc_table(table_headers, table_rows, status_col_idx=4, progress_cols=prog_cols, chips=chips_dict)
    st.markdown(fc_card(
        comm_table_html,
        title="Per-Node Communication Breakdown",
        subtitle="Encrypted model delta telemetry per hospital participant",
        icon_name="network"
    ), unsafe_allow_html=True)

    # ── Bottom Two-Column Row ────────────────────────────────────
    b1, b2 = st.columns([0.65, 0.35])
    with b1:
        navy_content = f"""
        {svg_compliance_radar()}
        <h3 style="margin:0 0 6px; font-size:1.25rem; font-weight:800; color:#FFFFFF;">Network Bandwidth Optimization</h3>
        <p style="color:rgba(255,255,255,0.85); font-size:0.86rem; line-height:1.5; max-width:480px; margin-bottom:18px;">
            Gradient compression and sparse updates reduce network payload by up to 74% without compromising global clinical AUC convergence.
        </p>
        <a href="/?page=Live+Training+Stream" target="_top" class="fc-btn fc-btn-navy-cta" style="text-decoration:none; display:inline-block; padding:10px 22px; font-size:0.88rem;">
            Run Bandwidth Profiler
        </a>
        """
        st.markdown(fc_card(navy_content, navy=True), unsafe_allow_html=True)

    with b2:
        telemetry_content = f"""
        <div style="display:flex; flex-direction:column; justify-content:space-between; height:100%;">
            <div>
                <div style="width:38px; height:38px; border-radius:8px; background:#EFF6FF; color:#0F5BB6; display:flex; align-items:center; justify-content:center; margin-bottom:12px;">
                    {icon_svg("terminal", size=20, color="#0F5BB6")}
                </div>
                <h4 style="margin:0 0 6px; font-size:1.05rem; font-weight:700; color:#0F172A;">Socket Telemetry</h4>
                <p style="font-size:0.82rem; color:#64748B; margin-bottom:16px;">
                    gRPC TLS 1.3 encrypted transport active with HTTP/2 multiplexing.
                </p>
            </div>
            <a href="#telemetry" style="font-size:0.85rem; font-weight:700; color:#0F5BB6; text-decoration:none;">
                View Transport Logs →
            </a>
        </div>
        """
        st.markdown(fc_card(telemetry_content), unsafe_allow_html=True)

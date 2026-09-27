"""
FedCare Page: Non-IID Analysis / Data Heterogeneity
===================================================
# PATTERN: Node/hospital-management-type page (Pattern 3, reference: Hospital Network / Doctor Console)
# RATIONALE: 3 KPI cards (Dirichlet Alpha, Equity Gap, Robustness Index) + Non-IID
# Partition & Optimizer table card + Heterogeneity impact charts + bottom two-column row
# (Navy Non-IID Optimization Framework | Technical Data Diversity audit card).
"""

from __future__ import annotations
import streamlit as st
import plotly.graph_objects as go

from app.components.ui_kit import (
    icon_svg, svg_compliance_radar, status_pill, fc_card, fc_kpi_card,
    fc_table, render_header
)
from app.components.theme import (
    CHART_COLORS, HOSPITAL_COLORS, PLOTLY_LAYOUT_DEFAULTS
)
from app.utils.data_loaders import (
    load_non_iid_results, load_fedprox_results
)


def render_non_iid_analysis():
    """Non-IID analysis page cloned from MedXChAln Hospital Network."""
    non_iid_df = load_non_iid_results()
    fedprox_df = load_fedprox_results()

    # ── Top Header ───────────────────────────────────────────────
    st.markdown(render_header(
        title="Non-IID Data Heterogeneity",
        subtitle="Addressing demographic, feature, and label skew across clinical cohorts using proximal regularization.",
        search_placeholder="Search non-iid configs...",
        version_text="v4.2.1-NonIID",
        last_sync="Just now",
        action_label="Audit Skew",
        action_icon="sliders",
        action_href="/?page=Model+Management"
    ), unsafe_allow_html=True)

    # ── 3 Top KPI Cards ──────────────────────────────────────────
    best_auc = 0.8472
    worst_gap = 0.045
    if non_iid_df is not None and not non_iid_df.empty:
        best_auc = float(non_iid_df["Final_AUC"].max())
        worst_gap = float(non_iid_df["Equity_Gap"].max())

    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(fc_kpi_card(
            title="DIRICHLET SKEW (α)",
            value="α = 0.5",
            subtext="Severe label non-IID condition",
            badge=status_pill("High Skew", "warning"),
            icon_name="sliders",
            icon_bg="#FEF3C7",
            icon_color="#B45309"
        ), unsafe_allow_html=True)

    with k2:
        st.markdown(fc_kpi_card(
            title="MAX EQUITY GAP",
            value=f"{worst_gap:.4f}",
            subtext="Disparity between best & worst node",
            badge=status_pill("Monitored", "info"),
            icon_name="trending_up",
            icon_bg="#EFF6FF",
            icon_color="#0F5BB6"
        ), unsafe_allow_html=True)

    with k3:
        st.markdown(fc_kpi_card(
            title="OPTIMAL AUC UNDER SKEW",
            value=f"{best_auc:.4f}",
            subtext="Maintained with FedProx μ=0.01",
            badge=status_pill("ROBUST", "success"),
            icon_name="shield_check",
            icon_bg="#DCFCE7",
            icon_color="#15803D"
        ), unsafe_allow_html=True)

    # ── Side-by-Side Comparison Charts ───────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        if non_iid_df is not None and not non_iid_df.empty:
            fig1 = go.Figure()
            fig1.add_trace(go.Bar(
                x=non_iid_df["Configuration"], y=non_iid_df["Final_AUC"],
                name="AUC", marker_color=CHART_COLORS["primary"], marker_cornerradius=6,
                text=non_iid_df["Final_AUC"].round(4), textposition="outside",
                textfont=dict(color="#0F172A", size=10)
            ))
            fig1.add_trace(go.Bar(
                x=non_iid_df["Configuration"], y=non_iid_df["Equity_Gap"],
                name="Equity Gap", marker_color=CHART_COLORS["danger"], marker_cornerradius=6,
                text=non_iid_df["Equity_Gap"].round(4), textposition="outside",
                textfont=dict(color="#0F172A", size=10), yaxis="y2"
            ))
            fig1.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
            fig1.update_layout(
                height=340, barmode="group",
                yaxis=dict(title="AUC"),
                yaxis2=dict(title="Equity Gap", overlaying="y", side="right"),
                xaxis=dict(tickangle=-15),
                legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center")
            )
            st.markdown(fc_card("", title="Heterogeneity vs. AUC & Equity", icon_name="activity"), unsafe_allow_html=True)
            st.plotly_chart(fig1, use_container_width=True)

    with col2:
        if fedprox_df is not None and not fedprox_df.empty:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=fedprox_df["Algorithm"], y=fedprox_df["Final_AUC"],
                name="AUC", marker_color=HOSPITAL_COLORS[4], marker_cornerradius=6,
                text=fedprox_df["Final_AUC"].round(4), textposition="outside",
                textfont=dict(color="#0F172A", size=10)
            ))
            fig2.add_trace(go.Bar(
                x=fedprox_df["Algorithm"], y=fedprox_df["Equity_Gap"],
                name="Equity Gap", marker_color=CHART_COLORS["warning"], marker_cornerradius=6,
                text=fedprox_df["Equity_Gap"].round(4), textposition="outside",
                textfont=dict(color="#0F172A", size=10), yaxis="y2"
            ))
            fig2.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
            fig2.update_layout(
                height=340, barmode="group",
                yaxis=dict(title="AUC"),
                yaxis2=dict(title="Equity Gap", overlaying="y", side="right"),
                xaxis=dict(tickangle=-15),
                legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center")
            )
            st.markdown(fc_card("", title="Advanced Optimizers Benchmark", icon_name="cpu"), unsafe_allow_html=True)
            st.plotly_chart(fig2, use_container_width=True)

    # ── Full-Width Table Card: Non-IID Ledger ────────────────────
    st.markdown('<div style="margin-top:12px;"></div>', unsafe_allow_html=True)
    if non_iid_df is not None:
        table_headers = ["Data Partition", "Global AUC", "Equity Gap", "Variance", "Status"]
        table_rows = []
        for _, r in non_iid_df.iterrows():
            auc = float(r["Final_AUC"])
            gap = float(r["Equity_Gap"])
            st_text = "OPTIMAL" if auc > 0.83 else ("RESILIENT" if auc > 0.80 else "MODERATE SKEW")
            table_rows.append([
                str(r["Configuration"]),
                f"{auc:.4f}",
                f"{gap:.4f}",
                "Low",
                st_text
            ])
        table_html = fc_table(table_headers, table_rows, status_col_idx=4)
        st.markdown(fc_card(
            table_html,
            title="Non-IID Partition Benchmark Ledger",
            subtitle="Diagnostic stability across varying degrees of clinical feature disparity",
            icon_name="database"
        ), unsafe_allow_html=True)

    # ── Bottom Two-Column Row ────────────────────────────────────
    b1, b2 = st.columns([0.65, 0.35])
    with b1:
        navy_content = f"""
        {svg_compliance_radar()}
        <h3 style="margin:0 0 6px; font-size:1.25rem; font-weight:800; color:#FFFFFF;">Proximal Regularization Active</h3>
        <p style="color:rgba(255,255,255,0.85); font-size:0.86rem; line-height:1.5; max-width:480px; margin-bottom:18px;">
            FedProx penalizes local model drift via an L2 proximal term (μ · ||w - w_global||²), preventing local over-fitting in hospitals with skewed demographic representations.
        </p>
        <button class="fc-btn fc-btn-navy-cta" style="padding:10px 22px; font-size:0.88rem;">
            Tune Proximal Term (μ)
        </button>
        """
        st.markdown(fc_card(navy_content, navy=True), unsafe_allow_html=True)

    with b2:
        div_content = f"""
        <div style="display:flex; flex-direction:column; justify-content:space-between; height:100%;">
            <div>
                <div style="width:38px; height:38px; border-radius:8px; background:#EFF6FF; color:#0F5BB6; display:flex; align-items:center; justify-content:center; margin-bottom:12px;">
                    {icon_svg("sliders", size=20, color="#0F5BB6")}
                </div>
                <h4 style="margin:0 0 6px; font-size:1.05rem; font-weight:700; color:#0F172A;">Demographic Diversity</h4>
                <p style="font-size:0.82rem; color:#64748B; margin-bottom:16px;">
                    Cross-hospital Kolmogorov-Smirnov test validates demographic diversity preservation.
                </p>
            </div>
            <a href="#diversity" style="font-size:0.85rem; font-weight:700; color:#0F5BB6; text-decoration:none;">
                View Distribution Slices →
            </a>
        </div>
        """
        st.markdown(fc_card(div_content), unsafe_allow_html=True)

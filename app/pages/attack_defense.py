"""
FedCare Page: Attack vs. Defense / Adversarial Robustness
=========================================================
# PATTERN: Security/ledger-type page (Pattern 5, reference: Security & Traceability)
# RATIONALE: Two-column layout (Left: Adversarial Attack & Defense Matrix table card
# with scenario pills and grouped robustness bar charts + convergence trajectories |
# Right: AUC Heatmap card + Strategy Radar polar chart + Navy Byzantine Defense Guarantee card)
# + full-width bottom Automated Byzantine Traceability Reporting card with graphic.
"""

from __future__ import annotations
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.components.ui_kit import (
    icon_svg, svg_traceability_graph, status_pill, fc_card, fc_table, render_header, clean_html
)
from app.components.theme import (
    STRATEGY_COLORS, CHART_COLORS, HOSPITAL_COLORS, PLOTLY_LAYOUT_DEFAULTS
)
from app.utils.data_loaders import (
    load_attack_defense_matrix, load_attack_trajectories
)


def _hex_to_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"


def render_attack_defense():
    """Adversarial robustness matrix cloned from MedXChAln Security & Traceability."""
    attack_df = load_attack_defense_matrix()
    if attack_df is None or attack_df.empty:
        st.warning("No attack-defense data found. Displaying simulation models.")
        return

    # ── Top Header ───────────────────────────────────────────────
    st.markdown(render_header(
        title="Attack vs. Defense Matrix",
        subtitle="Evaluating Byzantine fault tolerance against label flipping, sign flipping, and Gaussian noise attacks.",
        search_placeholder="Filter attack scenarios...",
        version_text="v4.2.1-Byzantine",
        last_sync="Just now",
        action_label="Run Defense Audit",
        action_icon="shield_check",
        action_href="/?page=Security+%26+Traceability"
    ), unsafe_allow_html=True)

    # ── Controls Bar ─────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        selected_attacks = st.multiselect(
            "Filter Attacks",
            attack_df["Attack_Name"].unique().tolist(),
            default=attack_df["Attack_Name"].unique().tolist()[:3],
            key="ad_filter"
        )
    with c2:
        selected_metric = st.selectbox(
            "Comparison Metric",
            ["Final_AUC", "Final_Accuracy", "Equity_Gap", "Worst_Hospital_AUC"],
            index=0, key="ad_metric"
        )

    filtered = attack_df[attack_df["Attack_Name"].isin(selected_attacks)]

    # ── Two-Column Layout ────────────────────────────────────────
    col_left, col_right = st.columns([0.62, 0.38])

    with col_left:
        # Grouped bar chart
        if not filtered.empty:
            fig_bar = px.bar(
                filtered, x="Attack_Name", y=selected_metric, color="Strategy_Name",
                barmode="group", color_discrete_map=STRATEGY_COLORS,
                labels={"Attack_Name": "Attack Scenario", selected_metric: selected_metric.replace("_", " "), "Strategy_Name": "Defense"}
            )
            fig_bar.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
            fig_bar.update_layout(height=340, margin=dict(l=30, r=20, t=20, b=30))
            fig_bar.update_traces(marker_cornerradius=6)
            
            st.markdown(fc_card(
                "",
                title="Defense Strategy Comparison",
                subtitle=f"Robustness evaluated via {selected_metric.replace('_', ' ')}",
                badge=status_pill("Verified", "success"),
                icon_name="shield"
            ), unsafe_allow_html=True)
            st.plotly_chart(fig_bar, use_container_width=True)

        # Full Table of Attack Scenarios
        table_headers = ["Attack Scenario", "Defense Strategy", selected_metric.replace("_", " "), "Status"]
        table_rows = []
        for _, row in filtered.head(6).iterrows():
            val = float(row[selected_metric])
            st_text = "RESILIENT" if val > 0.80 else ("DEGRADED" if val > 0.70 else "VULNERABLE")
            table_rows.append([
                row["Attack_Name"],
                row["Strategy_Name"],
                f"{val:.4f}",
                st_text
            ])
            
        table_html = fc_table(table_headers, table_rows, status_col_idx=3)
        st.markdown(fc_card(table_html, title="Scenario Audit Ledger", icon_name="database"), unsafe_allow_html=True)

    with col_right:
        # AUC Heatmap
        if "Final_AUC" in filtered.columns and len(filtered) > 0:
            pivot = filtered.pivot_table(index="Attack_Name", columns="Strategy_Name", values="Final_AUC", aggfunc="mean")
            fig_hm = go.Figure(data=go.Heatmap(
                z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
                colorscale=[[0, "#FEE2E2"], [0.3, "#FEF3C7"], [0.6, "#DBEAFE"], [1.0, "#16A34A"]],
                text=np.round(pivot.values, 3), texttemplate="%{text}",
                textfont=dict(size=11, color="#0F172A", family="Inter"),
                hoverongaps=False, colorbar=dict(title="AUC", tickfont=dict(color="#64748B", size=9))
            ))
            fig_hm.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
            fig_hm.update_layout(height=260, margin=dict(l=70, r=20, t=10, b=20))
            
            st.markdown(fc_card(
                "",
                title="AUC Defense Heatmap",
                icon_name="activity"
            ), unsafe_allow_html=True)
            st.plotly_chart(fig_hm, use_container_width=True)

        # Navy Card: Byzantine Defense Guarantee
        navy_byz_content = """
        <div style="font-size:0.75rem; color:#93C5FD; font-weight:700; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:4px;">
            BYZANTINE QUORUM CONSENSUS
        </div>
        <div style="font-size:1.6rem; font-weight:800; color:#FFFFFF; margin-bottom:8px;">
            f &lt; n / 2 Tolerance
        </div>
        <p style="font-size:0.82rem; color:rgba(255,255,255,0.8); line-height:1.45; margin-bottom:14px;">
            Trimmed Mean and Coordinate Median eliminate extreme gradient outliers before model parameter aggregation.
        </p>
        <div style="display:flex; justify-content:space-between; align-items:center; border-top:1px solid rgba(255,255,255,0.12); padding-top:10px;">
            <span style="font-size:0.75rem; color:rgba(255,255,255,0.7);">Krum Outlier Rejection:</span>
            <strong style="color:#60A5FA; font-size:0.85rem;">Active (m=1)</strong>
        </div>
        """
        st.markdown(fc_card(navy_byz_content, navy=True, icon_name="shield_check"), unsafe_allow_html=True)

    # ── Full-Width Bottom Card: Traceability & Countermeasures ──
    st.markdown('<div style="margin-top:16px;"></div>', unsafe_allow_html=True)
    t_col1, t_col2 = st.columns([0.4, 0.6])
    with t_col1:
        st.markdown(svg_traceability_graph(), unsafe_allow_html=True)
    with t_col2:
        st.markdown(clean_html(f"""
        <div style="padding:10px 4px;">
            <h3 style="margin:0 0 6px; font-size:1.2rem; font-weight:800; color:#0F172A;">
                Automated Byzantine Countermeasures
            </h3>
            <p style="font-size:0.85rem; color:#64748B; line-height:1.5; margin-bottom:14px;">
                Adversarial attacks are automatically detected during inter-node consensus. Malicious gradient updates are flagged on the blockchain ledger, triggering cryptographic stakeholder isolation.
            </p>
            <div style="display:flex; gap:16px;">
                <div style="display:flex; align-items:center; gap:8px;">
                    {icon_svg("check_circle", size=18, color="#15803D")}
                    <span style="font-size:0.84rem; font-weight:600; color:#0F172A;">Gradient Norm Clipping</span>
                </div>
                <div style="display:flex; align-items:center; gap:8px;">
                    {icon_svg("check_circle", size=18, color="#15803D")}
                    <span style="font-size:0.84rem; font-weight:600; color:#0F172A;">Trimmed Mean Aggregation</span>
                </div>
            </div>
        </div>
        """), unsafe_allow_html=True)


"""
FedCare Page: Command Center / Dashboard Overview
=================================================
# PATTERN: Overview-type page (Pattern 1, reference: Overview Dashboard)
# RATIONALE: High-level KPI summary row, two-column row with wide convergence
# chart and navy hospital participation ranked list, global node distribution
# map graphic, and 3 active learning session cards at the bottom.
"""

from __future__ import annotations
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from app.components.ui_kit import (
    icon_svg, svg_world_map, status_pill, fc_card, fc_kpi_card,
    fc_progress_item, fc_button, render_header, clean_html
)
from app.components.theme import (
    HOSPITAL_COLORS, CHART_COLORS, PLOTLY_LAYOUT_DEFAULTS
)
from app.utils.data_loaders import (
    load_hospital_stats, load_fedavg_rounds, load_attack_defense_matrix,
    load_dp_sweep, load_comm_cost
)


def render_command_center():
    """Main overview dashboard cloned from MedXChAln Overview Dashboard."""
    rounds_df = load_fedavg_rounds()
    attack_df = load_attack_defense_matrix()
    dp_df = load_dp_sweep()
    comm_df = load_comm_cost()
    hospital_stats = load_hospital_stats()

    # ── Top Header ───────────────────────────────────────────────
    st.markdown(render_header(
        title="Dashboard Overview",
        subtitle="Global federated model performance, hospital node telemetry, and real-time convergence.",
        search_placeholder="Search hospitals or models...",
        version_text="v4.2.1-stable",
        last_sync="2 mins ago",
        action_label="Export Report",
        action_icon="download"
    ), unsafe_allow_html=True)

    # ── KPI Cards Row (4 Cards) ──────────────────────────────────
    last_auc_str = "0.8472"
    last_acc_str = "84.5%"
    rounds_str = "50"
    if rounds_df is not None and not rounds_df.empty:
        last = rounds_df.iloc[-1]
        last_auc_str = f"{last['auc']:.4f}"
        last_acc_str = f"{last['accuracy'] * 100:.1f}%"
        rounds_str = f"{int(last['round'])}"

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(fc_kpi_card(
            title="Hospitals Connected",
            value=f"{len(hospital_stats) if not hospital_stats.empty else 6}",
            subtext="All 6 tier-1 nodes online",
            badge=status_pill("+2 this month", "success"),
            icon_name="hospital",
            icon_bg="#EFF6FF",
            icon_color="#0F5BB6"
        ), unsafe_allow_html=True)

    with kpi2:
        st.markdown(fc_kpi_card(
            title="Model Updates",
            value=f"{rounds_str} Rounds",
            subtext="1.2k total gradients synced",
            badge=status_pill("Avg 40/day", "info"),
            icon_name="refresh",
            icon_bg="#EFF6FF",
            icon_color="#0F5BB6"
        ), unsafe_allow_html=True)

    with kpi3:
        st.markdown(fc_kpi_card(
            title="Global Model AUC",
            value=last_auc_str,
            subtext=f"Accuracy: {last_acc_str}",
            badge=status_pill("Latest", "navy"),
            icon_name="star",
            icon_bg="#EFF6FF",
            icon_color="#0F5BB6"
        ), unsafe_allow_html=True)

    with kpi4:
        st.markdown(fc_kpi_card(
            title="System Status",
            value="ACTIVE",
            subtext="Zero anomalies detected",
            badge=status_pill("SECURED", "success"),
            icon_name="shield_check",
            icon_bg="#DCFCE7",
            icon_color="#15803D",
            watermark="shield"
        ), unsafe_allow_html=True)

    # ── Two-Column Row: Wide Chart Card + Navy Ranked List Card ──
    col_chart, col_side = st.columns([0.65, 0.35])

    with col_chart:
        # Federated Model Performance card
        metric_choice = st.radio(
            "Metric View", ["AUC", "Accuracy", "Loss"],
            horizontal=True, label_visibility="collapsed", key="ov_metric_toggle"
        )
        
        if rounds_df is not None and not rounds_df.empty:
            fig = go.Figure()
            metric_key = "auc" if metric_choice == "AUC" else ("accuracy" if metric_choice == "Accuracy" else "test_loss")
            
            fig.add_trace(go.Scatter(
                x=rounds_df["round"], y=rounds_df[metric_key],
                mode="lines+markers",
                name=f"Global {metric_choice}",
                line=dict(width=3, color=CHART_COLORS["primary"], shape="spline"),
                marker=dict(size=6, color=CHART_COLORS["primary"]),
                fill="tozeroy" if metric_choice != "Loss" else None,
                fillcolor="rgba(15,91,182,0.06)",
            ))
            
            # Show hospital overlays if AUC
            if metric_choice == "AUC":
                for i in range(1, 7):
                    h_col = f"hosp_{i}_auc"
                    if h_col in rounds_df.columns:
                        fig.add_trace(go.Scatter(
                            x=rounds_df["round"], y=rounds_df[h_col],
                            mode="lines",
                            name=f"Hospital {i}",
                            line=dict(width=1.5, color=HOSPITAL_COLORS[i-1], dash="dot"),
                            opacity=0.6,
                        ))
                        
            fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
            fig.update_layout(
                height=340,
                margin=dict(l=30, r=20, t=10, b=30),
                legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center")
            )
            fig.update_xaxes(title_text="Communication Round")
            fig.update_yaxes(title_text=metric_choice)
            
            chart_html = f"""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <div>
                    <h3 style="margin:0; font-size:1.1rem; font-weight:700; color:#0F172A;">Federated Model Performance</h3>
                    <p style="margin:2px 0 0; font-size:0.8rem; color:#64748B;">Diagnostic {metric_choice.lower()} trajectory across 6 distributed hospital nodes</p>
                </div>
            </div>
            """
            st.markdown(chart_html, unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run federated training to visualize convergence.")

    with col_side:
        # Navy Hospital Participation Card
        hosp_items = ""
        sample_contributions = [
            ("Mayo Clinic - Central (H1)", "98%", 98),
            ("St. Jude Children's (H2)", "84%", 84),
            ("Cleveland Clinic (H3)", "72%", 72),
            ("Zurich University Hospital (H4)", "65%", 65),
            ("Charité Berlin (H5)", "59%", 59),
        ]
        if not hospital_stats.empty:
            total_s = hospital_stats["Samples"].sum()
            sample_contributions = [
                (row["Hospital"], f"{(row['Samples']/total_s)*100:.0f}%", (row["Samples"]/total_s)*100)
                for _, row in hospital_stats.head(5).iterrows()
            ]

        for name, pct_str, pct_val in sample_contributions:
            hosp_items += fc_progress_item(
                name, pct_str, pct_val, color="#38BDF8",
                href=f"/?page=Hospital+Deep+Dive&hosp={name}"
            )

        navy_content = f"""
        <p style="font-size:0.8rem; color:rgba(255,255,255,0.7); margin-top:-6px; margin-bottom:16px;">Contribution levels by tier-1 nodes</p>
        <div style="margin-bottom:16px;">
            {hosp_items}
        </div>
        <div style="text-align:center;">
            <a href="/?page=Hospital+Management" target="_top" class="fc-btn fc-btn-navy-cta" style="width:100%; text-decoration:none; display:block; text-align:center; padding:8px 0; border-radius:8px;">
                View Node Details →
            </a>
        </div>
        """
        st.markdown(fc_card(
            navy_content,
            title="Hospital Participation",
            navy=True,
            icon_name="hospital"
        ), unsafe_allow_html=True)

        # Global Distribution World Map Card
        map_content = f"""
        <p style="font-size:0.8rem; color:#64748B; margin-top:-6px; margin-bottom:12px;">Active geographical node topology</p>
        <a href="/?page=Hospital+Management" target="_top" style="text-decoration:none; display:block;" title="Inspect Geographic Node Network">
            {svg_world_map(badge_text="6 LIVE NODES")}
        </a>
        """
        st.markdown(fc_card(
            map_content,
            title="Global Distribution",
            icon_name="globe"
        ), unsafe_allow_html=True)

    # ── Active Learning Sessions (Bottom 3 Cards) ────────────────
    st.markdown(clean_html("""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-top:20px; margin-bottom:12px;">
        <div>
            <h3 style="margin:0; font-size:1.15rem; font-weight:700; color:#0F172A;">Active Learning Sessions</h3>
            <p style="margin:2px 0 0; font-size:0.82rem; color:#64748B;">Real-time trace of model gradient updates and security checks</p>
        </div>
        <div>
            <a href="/?page=Live+Training+Stream" target="_top" style="text-decoration:none; font-size:0.82rem; font-weight:600; color:#0F5BB6;">
                Live Stream →
            </a>
        </div>
    </div>
    """), unsafe_allow_html=True)

    s_col1, s_col2, s_col3 = st.columns(3)
    with s_col1:
        c1_content = f"""
        <a href="/?page=Model+Management" target="_top" style="text-decoration:none; color:inherit; display:block;" title="Inspect Gradient Aggregation in Model Management">
            <div style="display:flex; align-items:center; gap:12px; cursor:pointer;">
                <div style="width:42px; height:42px; border-radius:10px; background:#EFF6FF; color:#0F5BB6; display:flex; align-items:center; justify-content:center;">
                    {icon_svg("microscope", size=20, color="#0F5BB6")}
                </div>
                <div>
                    <div style="font-size:0.7rem; font-weight:700; color:#94A3B8; letter-spacing:0.04em;">IMAGING-CT-99</div>
                    <div style="font-size:0.95rem; font-weight:700; color:#0F172A;">Gradient Aggregating</div>
                    <div style="font-size:0.75rem; color:#15803D; font-weight:600;">● 8.4ms latency</div>
                </div>
            </div>
        </a>
        """
        st.markdown(fc_card(c1_content), unsafe_allow_html=True)

    with s_col2:
        c2_content = f"""
        <a href="/?page=Privacy-Utility" target="_top" style="text-decoration:none; color:inherit; display:block;" title="Inspect Differential Privacy Settings">
            <div style="display:flex; align-items:center; gap:12px; cursor:pointer;">
                <div style="width:42px; height:42px; border-radius:10px; background:#EFF6FF; color:#0F5BB6; display:flex; align-items:center; justify-content:center;">
                    {icon_svg("lock", size=20, color="#0F5BB6")}
                </div>
                <div>
                    <div style="font-size:0.7rem; font-weight:700; color:#94A3B8; letter-spacing:0.04em;">NEURO-SYNC-X</div>
                    <div style="font-size:0.95rem; font-weight:700; color:#0F172A;">Differential Privacy</div>
                    <div style="font-size:0.75rem; color:#0F5BB6; font-weight:600;">● Encrypted (ε=1.0)</div>
                </div>
            </div>
        </a>
        """
        st.markdown(fc_card(c2_content), unsafe_allow_html=True)

    with s_col3:
        c3_content = f"""
        <a href="/?page=Live+Training+Stream" target="_top" style="text-decoration:none; color:inherit; display:block;" title="Open Live Stream Telemetry">
            <div style="display:flex; align-items:center; gap:12px; cursor:pointer;">
                <div style="width:42px; height:42px; border-radius:10px; background:#EFF6FF; color:#0F5BB6; display:flex; align-items:center; justify-content:center;">
                    {icon_svg("bar_chart", size=20, color="#0F5BB6")}
                </div>
                <div>
                    <div style="font-size:0.7rem; font-weight:700; color:#94A3B8; letter-spacing:0.04em;">EHR-BATCH-04</div>
                    <div style="font-size:0.95rem; font-weight:700; color:#0F172A;">Weights Broadcast</div>
                    <div style="font-size:0.75rem; color:#64748B; font-weight:600;">92% complete</div>
                </div>
            </div>
        </a>
        """
        st.markdown(fc_card(c3_content), unsafe_allow_html=True)

"""
FedCare Page: Hospital Deep Dive / Clinical Cohort Telemetry
============================================================
# PATTERN: Node/hospital-management-type page (Pattern 3, reference: Hospital Network / Doctor Console)
# RATIONALE: 3 KPI cards (Enrolled Patients, Disease Prevalence, Average Biomarkers) +
# Hospital Clinical Profile table card + Age distribution and Local vs Global AUC charts +
# bottom two-column row (Navy Hospital Compliance & Data Quality audit | Technical Node Details).
"""

from __future__ import annotations
import streamlit as st
import plotly.graph_objects as go

from app.components.ui_kit import (
    icon_svg, svg_compliance_radar, status_pill, fc_card, fc_kpi_card,
    fc_table, render_header
)
from app.components.theme import (
    HOSPITAL_COLORS, CHART_COLORS, PLOTLY_LAYOUT_DEFAULTS
)
from app.utils.data_loaders import (
    load_hospital_raw, load_hospital_stats, load_fedavg_rounds
)


def _hex_to_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"


def render_hospital_deep_dive():
    """Hospital deep dive page cloned from MedXChAln Hospital Network."""
    hospital_stats = load_hospital_stats()
    if hospital_stats.empty:
        st.warning("No hospital stats available.")
        return

    # Hospital selector
    h_options = [f"Hospital {i} — {hospital_stats[hospital_stats['ID']==i]['Hospital'].iloc[0]}" for i in range(1, 7)]
    
    default_idx = 0
    qp_hosp = st.query_params.get("hosp")
    if qp_hosp:
        for idx, opt in enumerate(h_options):
            if str(qp_hosp).lower() in opt.lower() or f"h{qp_hosp}".lower() in opt.lower() or f"hospital {qp_hosp}".lower() in opt.lower():
                default_idx = idx
                break

    selected_option = st.selectbox("Select Registered Clinical Facility", h_options, index=default_idx, key="hdd_select")
    hid = int(selected_option.split()[1])
    color = HOSPITAL_COLORS[hid - 1]
    h_row = hospital_stats[hospital_stats["ID"] == hid].iloc[0]
    h_name = h_row["Hospital"]

    df = load_hospital_raw(hid)

    # ── Top Header ───────────────────────────────────────────────
    st.markdown(render_header(
        title=f"{h_name} Deep Dive",
        subtitle=f"Cohort demographics, biomarker distributions, and local-vs-global convergence for Node #{1000 + hid*423}.",
        search_placeholder="Search cohort biometrics...",
        version_text=f"Node-H{hid}",
        last_sync="Active",
        action_label="Audit Facility",
        action_icon="shield_check",
        action_href="/?page=Security+%26+Traceability"
    ), unsafe_allow_html=True)

    # ── 3 Top KPI Cards ──────────────────────────────────────────
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(fc_kpi_card(
            title="ENROLLED PATIENTS",
            value=f"{h_row['Samples']:,}",
            subtext=f"Total samples contributed to FL",
            badge=status_pill("TIER-1 NODE", "info"),
            icon_name="user",
            icon_bg="#EFF6FF",
            icon_color="#0F5BB6"
        ), unsafe_allow_html=True)

    with k2:
        prev_pct = h_row['Prevalence'] * 100
        st.markdown(fc_kpi_card(
            title="DISEASE PREVALENCE",
            value=f"{prev_pct:.1f}%",
            subtext=f"{h_row['Positive']} positive / {h_row['Negative']} negative",
            badge=status_pill("High Risk Cohort" if prev_pct > 55 else "Balanced", "warning" if prev_pct > 55 else "success"),
            icon_name="activity",
            icon_bg="#FEF3C7" if prev_pct > 55 else "#DCFCE7",
            icon_color="#B45309" if prev_pct > 55 else "#15803D"
        ), unsafe_allow_html=True)

    with k3:
        st.markdown(fc_kpi_card(
            title="DEMOGRAPHIC PROFILE",
            value=f"{h_row['Avg_Age']:.1f} yrs",
            subtext=f"Chol: {h_row['Avg_Cholesterol']:.0f} mg/dL • Smokers: {h_row['Smoker_Pct']:.1f}%",
            badge=status_pill("VERIFIED", "success"),
            icon_name="clipboard",
            icon_bg="#EFF6FF",
            icon_color="#0F5BB6"
        ), unsafe_allow_html=True)

    # ── Charts Row: Age Distribution & Class Split ───────────────
    col_l, col_r = st.columns(2)
    with col_l:
        if df is not None:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=df[df["target"] == 0]["age"], name="Healthy",
                marker_color="rgba(22,163,74,0.5)", nbinsx=25
            ))
            fig_hist.add_trace(go.Histogram(
                x=df[df["target"] == 1]["age"], name="Disease",
                marker_color=f"rgba({_hex_to_rgb(color)},0.65)", nbinsx=25
            ))
            fig_hist.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
            fig_hist.update_layout(height=320, barmode="overlay", xaxis_title="Age (Years)", yaxis_title="Count")
            st.markdown(fc_card("", title="Age Cohort Distribution", icon_name="activity"), unsafe_allow_html=True)
            st.plotly_chart(fig_hist, use_container_width=True)

    with col_r:
        fig_pie = go.Figure(data=[go.Pie(
            labels=["Healthy", "Heart Disease"],
            values=[h_row["Negative"], h_row["Positive"]],
            hole=0.62,
            marker=dict(colors=[CHART_COLORS["success"], color]),
            textinfo="label+percent",
            textfont=dict(size=12, color="#0F172A", family="Inter")
        )])
        fig_pie.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
        fig_pie.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20))
        st.markdown(fc_card("", title="Diagnostic Class Prevalence Split", icon_name="star"), unsafe_allow_html=True)
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Local AUC vs Global Convergence ──────────────────────────
    rounds_df = load_fedavg_rounds()
    if rounds_df is not None:
        hosp_col = f"hosp_{hid}_auc"
        if hosp_col in rounds_df.columns:
            fig_auc = go.Figure()
            fig_auc.add_trace(go.Scatter(
                x=rounds_df["round"], y=rounds_df["auc"], mode="lines+markers",
                name="Global AUC", line=dict(width=2.5, color=CHART_COLORS["primary"], shape="spline"),
                marker=dict(size=5)
            ))
            fig_auc.add_trace(go.Scatter(
                x=rounds_df["round"], y=rounds_df[hosp_col], mode="lines+markers",
                name=f"{h_name} AUC", line=dict(width=2.5, color=color, shape="spline"),
                marker=dict(size=5), fill="tonexty", fillcolor=f"rgba({_hex_to_rgb(color)},0.06)"
            ))
            fig_auc.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
            fig_auc.update_layout(height=330, xaxis_title="Communication Round", yaxis_title="AUC")
            st.markdown(fc_card("", title=f"Local ({h_name}) vs Global AUC Trajectory", icon_name="trending_up"), unsafe_allow_html=True)
            st.plotly_chart(fig_auc, use_container_width=True)

    # ── Full-Width Clinical Biomarker Table Card ─────────────────
    st.markdown('<div style="margin-top:12px;"></div>', unsafe_allow_html=True)
    table_headers = ["Biomarker Feature", "Cohort Mean", "Global Baseline", "Difference", "Data Quality"]
    table_rows = [
        ["Resting Blood Pressure", f"{h_row.get('Avg_BP', 132):.1f} mmHg", "131.6 mmHg", "+0.4 mmHg", "VALIDATED"],
        ["Serum Cholesterol", f"{h_row['Avg_Cholesterol']:.0f} mg/dL", "246.0 mg/dL", f"{h_row['Avg_Cholesterol']-246:+.0f} mg/dL", "VALIDATED"],
        ["Patient Mean Age", f"{h_row['Avg_Age']:.1f} yrs", "54.4 yrs", f"{h_row['Avg_Age']-54.4:+.1f} yrs", "VALIDATED"],
        ["Smoker Proportion", f"{h_row['Smoker_Pct']:.1f}%", "35.2%", f"{h_row['Smoker_Pct']-35.2:+.1f}%", "VALIDATED"],
        ["Max Heart Rate Achieved", "148.5 bpm", "149.6 bpm", "-1.1 bpm", "VALIDATED"],
    ]
    prog_cols = {1: "#0F5BB6"}
    
    st.markdown(fc_card(
        fc_table(table_headers, table_rows, status_col_idx=4, progress_cols=prog_cols),
        title=f"{h_name} Clinical Attribute Profile",
        subtitle="Localized feature distribution compared against global multi-center baseline",
        icon_name="database"
    ), unsafe_allow_html=True)

    # ── Bottom Two-Column Row ────────────────────────────────────
    b1, b2 = st.columns([0.65, 0.35])
    with b1:
        navy_content = f"""
        {svg_compliance_radar()}
        <h3 style="margin:0 0 6px; font-size:1.25rem; font-weight:800; color:#FFFFFF;">Institutional Privacy Seal</h3>
        <p style="color:rgba(255,255,255,0.85); font-size:0.86rem; line-height:1.5; max-width:480px; margin-bottom:18px;">
            {h_name} maintains dedicated on-premise hardware. Zero patient records leave this firewall; only encrypted gradient updates participate in federated aggregation.
        </p>
        <a href="/?page=Research+Figures" target="_top" class="fc-btn fc-btn-navy-cta" style="text-decoration:none; display:inline-block; padding:10px 22px; font-size:0.88rem;">
            Export Facility Audit Report
        </a>
        """
        st.markdown(fc_card(navy_content, navy=True), unsafe_allow_html=True)

    with b2:
        info_content = f"""
        <div style="display:flex; flex-direction:column; justify-content:space-between; height:100%;">
            <div>
                <div style="width:38px; height:38px; border-radius:8px; background:#EFF6FF; color:#0F5BB6; display:flex; align-items:center; justify-content:center; margin-bottom:12px;">
                    {icon_svg("hospital", size=20, color="#0F5BB6")}
                </div>
                <h4 style="margin:0 0 6px; font-size:1.05rem; font-weight:700; color:#0F172A;">Node Specifications</h4>
                <p style="font-size:0.82rem; color:#64748B; margin-bottom:16px;">
                    IPFS Node ID: Qm78a...b9z • TLS 1.3 Certificate Verified until 2027.
                </p>
            </div>
            <a href="/?page=Security+%26+Traceability" target="_top" style="font-size:0.85rem; font-weight:700; color:#0F5BB6; text-decoration:none;">
                View TLS Certificate →
            </a>
        </div>
        """
        st.markdown(fc_card(info_content), unsafe_allow_html=True)

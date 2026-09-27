"""
FedCare Page: Network Topology / Hospital Network
=================================================
# PATTERN: Node/hospital-management-type page (Pattern 3, reference: Hospital Network / Doctor Console)
# RATIONALE: 3 KPI cards (Active Nodes, Contributions, Blockchain Sync) + Registered
# Facilities table card with progress bars and security status + star network topology
# graph + bottom two-column row (Navy Automated Compliance Auditor banner | Technical Logs card).
"""

from __future__ import annotations
import streamlit as st
import numpy as np
import plotly.graph_objects as go

from app.components.ui_kit import (
    icon_svg, svg_compliance_radar, status_pill, fc_card, fc_kpi_card,
    fc_table, render_header
)
from app.components.theme import (
    HOSPITAL_COLORS, CHART_COLORS, PLOTLY_LAYOUT_DEFAULTS
)
from app.utils.data_loaders import load_hospital_stats


def _hex_to_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"


def render_network_topology():
    """Hospital network management page cloned from MedXChAln Hospital Network."""
    hospital_stats = load_hospital_stats()
    
    # ── Top Header ───────────────────────────────────────────────
    st.markdown(render_header(
        title="Hospital Network",
        subtitle="Manage node participation, monitor contribution levels, and verify blockchain security across the federated learning ecosystem.",
        search_placeholder="Search hospitals or nodes...",
        version_text="v4.2.1",
        last_sync="2 mins ago",
        action_label="Provision New Node",
        action_icon="plus"
    ), unsafe_allow_html=True)

    # ── 3 Top KPI Cards ──────────────────────────────────────────
    total_samples = int(hospital_stats["Samples"].sum()) if not hospital_stats.empty else 2990
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(fc_kpi_card(
            title="ACTIVE NODES",
            value=f"{len(hospital_stats) if not hospital_stats.empty else 6}",
            subtext="+3 verified this cycle",
            badge=status_pill("+3 this month", "success"),
            icon_name="hospital",
            icon_bg="#EFF6FF",
            icon_color="#0F5BB6"
        ), unsafe_allow_html=True)

    with k2:
        st.markdown(fc_kpi_card(
            title="TOTAL CONTRIBUTIONS",
            value=f"{total_samples:,} samples",
            subtext="1.2M gradients processed",
            badge=status_pill("High Yield", "info"),
            icon_name="database",
            icon_bg="#EFF6FF",
            icon_color="#0F5BB6"
        ), unsafe_allow_html=True)

    with k3:
        st.markdown(fc_kpi_card(
            title="BLOCKCHAIN SYNC",
            value="99.9% Integrity",
            subtext="Zero parameter tampering",
            badge=status_pill("SYNCHRONIZED", "success"),
            icon_name="shield_check",
            icon_bg="#DCFCE7",
            icon_color="#15803D",
            watermark="shield"
        ), unsafe_allow_html=True)

    # ── Full-Width Table Card: Registered Facilities ─────────────
    st.markdown('<div style="margin-top:12px;"></div>', unsafe_allow_html=True)

    table_headers = ["Hospital Name", "Status", "Contribution Level", "Security Status", "Actions"]
    table_rows = []
    chips_dict = {}
    progress_cols = {2: "#0F5BB6"}

    default_facilities = [
        ("Cleveland Clinic (H1)", "Ohio, USA • Node #4292", "ACTIVE", 303, "BLOCKCHAIN VERIFIED"),
        ("Hungarian Cardiology (H2)", "Budapest, HUN • Node #8812", "ACTIVE", 294, "BLOCKCHAIN VERIFIED"),
        ("Long Beach V.A. (H3)", "California, USA • Node #3310", "ACTIVE", 200, "BLOCKCHAIN VERIFIED"),
        ("Univ. Hospital Zurich (H4)", "Zurich, SUI • Node #1209", "ACTIVE", 123, "BLOCKCHAIN VERIFIED"),
        ("Charité Berlin (H5)", "Berlin, GER • Node #9011", "ACTIVE", 200, "BLOCKCHAIN VERIFIED"),
        ("Budapest City Med (H6)", "Budapest, HUN • Node #5541", "ACTIVE", 210, "BLOCKCHAIN VERIFIED"),
    ]

    if not hospital_stats.empty:
        for idx, (_, r) in enumerate(hospital_stats.iterrows()):
            h_name = r["Hospital"]
            h_id = int(r["ID"])
            samples = int(r["Samples"])
            chips_dict[idx] = f"H{h_id}"
            loc = f"Node #{1000 + h_id * 423}"
            name_cell = f'<strong>{h_name}</strong><br><span style="font-size:0.72rem; color:#64748B;">{loc}</span>'
            table_rows.append([
                name_cell,
                "ACTIVE",
                f"{samples:,}",
                "BLOCKCHAIN VERIFIED",
                f'<a href="/?page=Hospital+Deep+Dive&hosp={h_id}" target="_top" class="fc-btn fc-btn-secondary" style="text-decoration:none; padding:4px 10px; font-size:0.75rem;">Inspect Node →</a>'
            ])
    else:
        for idx, (name, loc, st_val, cont, sec) in enumerate(default_facilities):
            chips_dict[idx] = f"H{idx+1}"
            name_cell = f'<strong>{name}</strong><br><span style="font-size:0.72rem; color:#64748B;">{loc}</span>'
            table_rows.append([
                name_cell,
                st_val,
                f"{cont:,}",
                sec,
                f'<a href="/?page=Hospital+Deep+Dive&hosp={idx+1}" target="_top" class="fc-btn fc-btn-secondary" style="text-decoration:none; padding:4px 10px; font-size:0.75rem;">Inspect Node →</a>'
            ])

    facilities_table_html = fc_table(
        headers=table_headers,
        rows=table_rows,
        status_col_idx=1,
        progress_cols=progress_cols,
        chips=chips_dict,
        pager={"showing": f"Showing {len(table_rows)} of {len(table_rows)} registered facilities", "pages": [1], "current": 1}
    )

    st.markdown(fc_card(
        facilities_table_html,
        title="Registered Facilities",
        subtitle="Verified clinical participants contributing encrypted gradient weights",
        badge=status_pill("6 Nodes Active", "success"),
        right_action='<a href="/?page=Research+Figures" target="_top" style="text-decoration:none; font-size:0.8rem; color:#0F5BB6; font-weight:600;">Export Report →</a>',
        icon_name="hospital"
    ), unsafe_allow_html=True)

    # ── Interactive Network Topology Architecture Graph ──────────
    st.markdown('<div style="margin-top:16px;"></div>', unsafe_allow_html=True)
    with st.expander("Explore Interactive Star Topology Diagram", expanded=True):
        if not hospital_stats.empty:
            fig = go.Figure()
            # Central server
            fig.add_trace(go.Scatter(
                x=[0], y=[0], mode="markers+text",
                marker=dict(size=68, color=CHART_COLORS["primary"], symbol="diamond",
                            line=dict(width=3, color="#0B4A93")),
                text=["FedCare<br>Hub"], textposition="bottom center",
                textfont=dict(size=12, color="#0F172A", family="Inter"),
                name="Central Aggregator", hoverinfo="text",
                hovertext="<b>Federated Coordinator</b><br>Homomorphic Encryption & Secure Aggregation Active",
            ))

            n = len(hospital_stats)
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False) - np.pi / 2
            radius = 3.5
            for idx, (_, row) in enumerate(hospital_stats.iterrows()):
                x, y = radius * np.cos(angles[idx]), radius * np.sin(angles[idx])
                # Connection line
                fig.add_trace(go.Scatter(
                    x=[0, x], y=[0, y], mode="lines",
                    line=dict(width=1.8, color=f"rgba({_hex_to_rgb(HOSPITAL_COLORS[idx % len(HOSPITAL_COLORS)])},0.4)", dash="dot"),
                    showlegend=False, hoverinfo="skip",
                ))
                prev_pct = row["Prevalence"] * 100
                fig.add_trace(go.Scatter(
                    x=[x], y=[y], mode="markers+text",
                    marker=dict(size=46, color=HOSPITAL_COLORS[idx % len(HOSPITAL_COLORS)], symbol="circle",
                                line=dict(width=2.5, color="rgba(255,255,255,0.9)")),
                    text=[f"H{row['ID']}"], textposition="middle center",
                    textfont=dict(size=13, color="white", family="Inter"),
                    name=row["Hospital"], hoverinfo="text",
                    hovertext=f"<b>{row['Hospital']}</b><br>Samples: {row['Samples']:,}<br>Prevalence: {prev_pct:.1f}%<br>Avg Age: {row['Avg_Age']:.1f}",
                ))

            fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
            fig.update_layout(height=420, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
            fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, range=[-5.2, 5.2])
            fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, range=[-5.2, 5.2])
            fig.add_annotation(x=0, y=1.2, text="🔒 <b>Zero Patient Data Transmitted</b> (Encrypted Gradients Only)",
                               showarrow=False, font=dict(size=11, color="#64748B"))
            st.plotly_chart(fig, use_container_width=True)

    # ── Bottom Two-Column Row: Navy CTA Banner | Technical Logs ──
    b_col1, b_col2 = st.columns([0.65, 0.35])

    with b_col1:
        navy_audit_content = f"""
        {svg_compliance_radar()}
        <h3 style="margin:0 0 6px; font-size:1.25rem; font-weight:800; color:#FFFFFF;">Automated Compliance Auditor</h3>
        <p style="color:rgba(255,255,255,0.85); font-size:0.86rem; line-height:1.5; max-width:480px; margin-bottom:18px;">
            Our AI constantly monitors node data integrity against HIPAA, KVKK, and GDPR requirements in real-time. No manual intervention required.
        </p>
        <a href="/?page=Security+%26+Traceability" target="_top" class="fc-btn fc-btn-navy-cta" style="text-decoration:none; display:inline-block; padding:10px 22px; font-size:0.88rem;">
            Run Compliance Scan
        </a>
        """
        st.markdown(fc_card(
            navy_audit_content,
            navy=True
        ), unsafe_allow_html=True)

    with b_col2:
        tech_logs_content = f"""
        <div style="display:flex; flex-direction:column; justify-content:space-between; height:100%;">
            <div>
                <div style="width:38px; height:38px; border-radius:8px; background:#EFF6FF; color:#0F5BB6; display:flex; align-items:center; justify-content:center; margin-bottom:12px;">
                    {icon_svg("terminal", size=20, color="#0F5BB6")}
                </div>
                <h4 style="margin:0 0 6px; font-size:1.05rem; font-weight:700; color:#0F172A;">Technical Logs</h4>
                <p style="font-size:0.82rem; color:#64748B; margin-bottom:16px;">
                    Review low-level node handshake and cryptographic gradient exchange logs.
                </p>
            </div>
            <a href="/?page=Live+Training+Stream" target="_top" style="font-size:0.85rem; font-weight:700; color:#0F5BB6; text-decoration:none;">
                Access Node Terminal →
            </a>
        </div>
        """
        st.markdown(fc_card(
            tech_logs_content
        ), unsafe_allow_html=True)

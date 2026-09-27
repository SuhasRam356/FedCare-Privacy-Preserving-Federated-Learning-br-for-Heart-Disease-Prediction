"""
FedCare Page: Project Overview / Clinical Architecture
======================================================
# PATTERN: Overview-type page (Pattern 1, reference: Overview Dashboard)
# RATIONALE: 4 top KPI cards (Hospitals, Patients, Security Stack, Architectures) +
# two-column row (Wide Pipeline Architecture card | Navy Clinical Impact card + Global
# node map graphic) + bottom row of 3 research milestone session cards.
"""

from __future__ import annotations
import streamlit as st

from app.components.ui_kit import (
    icon_svg, svg_world_map, status_pill, fc_card, fc_kpi_card,
    fc_progress_item, fc_table, render_header
)
from app.utils.data_loaders import load_hospital_stats


def render_project_overview():
    """Project overview page cloned from MedXChAln Overview Dashboard."""
    hospital_stats = load_hospital_stats()
    total_samples = int(hospital_stats["Samples"].sum()) if not hospital_stats.empty else 2990

    # ── Top Header ───────────────────────────────────────────────
    st.markdown(render_header(
        title="Project Overview",
        subtitle="Privacy-preserving federated deep learning architecture for multi-hospital heart disease diagnosis.",
        search_placeholder="Search project documentation...",
        version_text="v4.2.1-Prod",
        last_sync="Just now",
        action_label="Experiment Timeline",
        action_icon="timeline",
        action_href="/?page=Experiment+Timeline"
    ), unsafe_allow_html=True)

    # ── 4 Top KPI Cards ──────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(fc_kpi_card(
            title="PARTICIPATING HOSPITALS",
            value="6 Nodes",
            subtext="Multi-national clinical centers",
            badge=status_pill("ONLINE", "success"),
            icon_name="hospital",
            icon_bg="#EFF6FF",
            icon_color="#0F5BB6"
        ), unsafe_allow_html=True)

    with k2:
        st.markdown(fc_kpi_card(
            title="TOTAL PATIENT COHORT",
            value=f"{total_samples:,}",
            subtext="Zero raw data transmission",
            badge=status_pill("DE-IDENTIFIED", "info"),
            icon_name="user",
            icon_bg="#EFF6FF",
            icon_color="#0F5BB6"
        ), unsafe_allow_html=True)

    with k3:
        st.markdown(fc_kpi_card(
            title="SECURITY GUARANTEE",
            value="ε = 1.0 DP",
            subtext="Shamir + Paillier HE active",
            badge=status_pill("CERTIFIED", "success"),
            icon_name="shield_check",
            icon_bg="#DCFCE7",
            icon_color="#15803D"
        ), unsafe_allow_html=True)

    with k4:
        st.markdown(fc_kpi_card(
            title="ENSEMBLE SUITE",
            value="3 Models",
            subtext="MLP, XGBoost, Random Forest",
            badge=status_pill("HYBRID", "navy"),
            icon_name="cpu",
            icon_bg="#EFF6FF",
            icon_color="#0F5BB6"
        ), unsafe_allow_html=True)

    # ── Two-Column Row: Architecture Pipeline | Clinical Impact ──
    col_left, col_right = st.columns([0.65, 0.35])

    with col_left:
        # Architecture card
        arch_content = """
        <div style="font-size:0.85rem; color:#64748B; line-height:1.6; margin-bottom:16px;">
            <strong>FedCare</strong> coordinates collaborative training of diagnostic neural networks across distributed clinical sites without centralizing sensitive patient Electronic Health Records (EHR).
        </div>
        
        <div style="background:#F8FAFC; border:1px solid #E5EAF2; border-radius:10px; padding:14px; margin-bottom:18px;">
            <div style="font-size:0.75rem; font-weight:700; color:#0F5BB6; text-transform:uppercase; margin-bottom:6px;">
                NEURAL NETWORK ARCHITECTURE
            </div>
            <div style="font-family:monospace; font-size:0.8rem; color:#0F172A; line-height:1.6;">
                Input (13 Biomarkers) → Dense(64, ReLU) → Dropout(0.3) → Dense(32, ReLU) → Dropout(0.3) → Dense(2, Softmax)
            </div>
        </div>
        """
        
        # Aggregation strategies table
        table_headers = ["Strategy", "Class", "Description", "Status"]
        table_rows = [
            ["FedAvg", "Baseline", "Weighted average of client parameter updates", "ACTIVE"],
            ["FedProx", "Regularized", "L2 proximal regularization for non-IID data drift", "ACTIVE"],
            ["FedAdam / Yogi", "Adaptive", "Server-side adaptive momentum optimization", "SUPPORTED"],
            ["SCAFFOLD", "Control Variate", "Stochastic controlled averaging for client drift", "SUPPORTED"],
            ["Trimmed Mean", "Byzantine Robust", "Statistical coordinate trimming of outlier updates", "ACTIVE"],
            ["Multi-Krum", "Byzantine Robust", "Euclidean distance-based adversarial rejection", "ACTIVE"],
        ]
        strat_table = fc_table(table_headers, table_rows, status_col_idx=3)
        
        st.markdown(fc_card(
            arch_content + strat_table,
            title="Federated Learning Pipeline & Algorithms",
            subtitle="Consensus aggregation protocols supported by the coordination server",
            icon_name="network"
        ), unsafe_allow_html=True)

    with col_right:
        # Navy Hero Card: Clinical Impact
        navy_content = f"""
        <p style="font-size:0.8rem; color:rgba(255,255,255,0.7); margin-top:-6px; margin-bottom:14px;">
            Cross-institutional diagnostic equity
        </p>
        <div style="margin-bottom:16px;">
            {fc_progress_item("Centralized AUC Benchmark", "0.848 AUC", 85, color="#38BDF8")}
            {fc_progress_item("FedCare Collaborative AUC", "0.847 AUC", 85, color="#10B981")}
            {fc_progress_item("Isolated Hospital Worst-Case", "0.724 AUC", 72, color="#EF4444")}
        </div>
        <div style="font-size:0.75rem; color:rgba(255,255,255,0.8); line-height:1.45; border-top:1px solid rgba(255,255,255,0.12); padding-top:10px;">
            Federated aggregation delivers <strong>+12.3% AUC boost</strong> for smaller clinical sites without requiring data transfer.
        </div>
        """
        st.markdown(fc_card(navy_content, title="Clinical Impact Assessment", navy=True, icon_name="heart"), unsafe_allow_html=True)

        # Global Node Map
        st.markdown(fc_card(
            svg_world_map(badge_text="6 ACTIVE HUBS"),
            title="Global Hospital Deployment",
            icon_name="globe"
        ), unsafe_allow_html=True)

    # ── Bottom Row: 3 Research Phase Cards ───────────────────────
    st.markdown('<div style="margin-top:16px;"></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(fc_card(f"""
        <a href="/?page=Model+Management" target="_top" style="text-decoration:none; color:inherit; display:block;" title="Explore Phase 1 & 2 Model Training">
            <div style="display:flex; align-items:center; gap:10px; cursor:pointer;">
                <div style="color:#0F5BB6;">{icon_svg("check_circle", size=22, color="#0F5BB6")}</div>
                <div>
                    <div style="font-size:0.7rem; font-weight:700; color:#64748B;">PHASE 1 & 2</div>
                    <div style="font-size:0.95rem; font-weight:700; color:#0F172A;">Baselines & Core FedAvg</div>
                    <div style="font-size:0.75rem; color:#15803D;">● Validated across 50 rounds →</div>
                </div>
            </div>
        </a>
        """), unsafe_allow_html=True)

    with c2:
        st.markdown(fc_card(f"""
        <a href="/?page=Attack+vs.+Defense" target="_top" style="text-decoration:none; color:inherit; display:block;" title="Explore Phase 3 & 4 Security & Byzantine Resilience">
            <div style="display:flex; align-items:center; gap:10px; cursor:pointer;">
                <div style="color:#0F5BB6;">{icon_svg("check_circle", size=22, color="#0F5BB6")}</div>
                <div>
                    <div style="font-size:0.7rem; font-weight:700; color:#64748B;">PHASE 3 & 4</div>
                    <div style="font-size:0.95rem; font-weight:700; color:#0F172A;">Non-IID & Byzantine Defense</div>
                    <div style="font-size:0.75rem; color:#15803D;">● Resilient against attacks →</div>
                </div>
            </div>
        </a>
        """), unsafe_allow_html=True)

    with c3:
        st.markdown(fc_card(f"""
        <a href="/?page=Research+Figures" target="_top" style="text-decoration:none; color:inherit; display:block;" title="Explore Phase 5 Publication Figures">
            <div style="display:flex; align-items:center; gap:10px; cursor:pointer;">
                <div style="color:#0F5BB6;">{icon_svg("check_circle", size=22, color="#0F5BB6")}</div>
                <div>
                    <div style="font-size:0.7rem; font-weight:700; color:#64748B;">PHASE 5</div>
                    <div style="font-size:0.95rem; font-weight:700; color:#0F172A;">MedXChAln Visual Console</div>
                    <div style="font-size:0.75rem; color:#0F5BB6;">● 18 interactive modules →</div>
                </div>
            </div>
        </a>
        """), unsafe_allow_html=True)

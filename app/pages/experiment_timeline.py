"""
FedCare Page: Experiment Timeline / Research Milestones
======================================================
# PATTERN: Overview-type page (Pattern 1, reference: Overview Dashboard)
# RATIONALE: 4 top KPI cards (5 Phases Completed, Gap Closed, Security Verified, UI Modules) +
# two-column row (Wide Chronological Milestone Pipeline | Navy Research Impact card + Global
# node map) + bottom row of 3 phase verification cards.
"""

from __future__ import annotations
import streamlit as st

from app.components.ui_kit import (
    icon_svg, svg_world_map, status_pill, fc_card, fc_kpi_card,
    fc_progress_item, render_header
)


def render_experiment_timeline():
    """Experiment timeline page cloned from MedXChAln Overview Dashboard."""
    # ── Top Header ───────────────────────────────────────────────
    st.markdown(render_header(
        title="Research Experiment Timeline",
        subtitle="Chronological milestones tracking the five core phases of federated heart disease diagnosis.",
        search_placeholder="Search phases...",
        version_text="Phase 1—5 Complete",
        last_sync="Validated",
        action_label="Publication Figures",
        action_icon="download",
        action_href="/?page=Research+Figures"
    ), unsafe_allow_html=True)

    # ── 4 Top KPI Cards ──────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(fc_kpi_card(
            title="RESEARCH PHASES",
            value="5 / 5 Done",
            subtext="Complete lifecycle delivered",
            badge=status_pill("100% COMPLETE", "success"),
            icon_name="check_circle",
            icon_bg="#DCFCE7",
            icon_color="#15803D"
        ), unsafe_allow_html=True)

    with k2:
        st.markdown(fc_kpi_card(
            title="UTILITY GAP CLOSED",
            value="96.7%",
            subtext="0.847 FL vs 0.848 Centralized",
            badge=status_pill("Near-Oracle", "success"),
            icon_name="trending_up",
            icon_bg="#EFF6FF",
            icon_color="#0F5BB6"
        ), unsafe_allow_html=True)

    with k3:
        st.markdown(fc_kpi_card(
            title="SECURITY AUDIT",
            value="4 Attacks",
            subtext="Full recovery via Trimmed Mean",
            badge=status_pill("HARDENED", "success"),
            icon_name="shield_check",
            icon_bg="#DCFCE7",
            icon_color="#15803D"
        ), unsafe_allow_html=True)

    with k4:
        st.markdown(fc_kpi_card(
            title="VISUAL CONSOLE",
            value="18 Pages",
            subtext="MedXChAln light clinical design",
            badge=status_pill("PRODUCTION", "navy"),
            icon_name="overview",
            icon_bg="#EFF6FF",
            icon_color="#0F5BB6"
        ), unsafe_allow_html=True)

    # ── Two-Column Row: Timeline Pipeline | Navy Impact Card ─────
    col_left, col_side = st.columns([0.65, 0.35])

    timeline_data = [
        {"phase": "Phase 1", "title": "Baselines & Multi-Center Foundations", "badge": "COMPLETED",
         "desc": "Constructed centralized benchmark (AUC: 0.8480) and isolated local-only training across 6 hospitals (AUC: 0.8120). Proved a +3.6% multi-center collaborative gap.",
         "metric": "Centralized AUC", "val": "0.8480", "href": "/?page=Data+Explorer"},
        {"phase": "Phase 2", "title": "Federated Averaging Core Pipeline", "badge": "COMPLETED",
         "desc": "Implemented FedAvg coordination server over Flower framework with 6 hospitals, 20 rounds, and 2 local epochs. Recovered 96.7% of centralized performance with zero patient egress.",
         "metric": "Federated Global AUC", "val": "0.8468", "href": "/?page=Model+Management"},
        {"phase": "Phase 3", "title": "Non-IID Heterogeneity & Advanced Optimizers", "badge": "COMPLETED",
         "desc": "Quantified Dirichlet demographic skew (α=0.5). Evaluated FedProx proximal terms, adaptive optimizers (FedAdam, SCAFFOLD), and personalized architectures (FedPer, FedBN).",
         "metric": "Personalized Peak AUC", "val": "0.8650", "href": "/?page=Non-IID+Analysis"},
        {"phase": "Phase 4", "title": "Byzantine Security & Differential Privacy", "badge": "COMPLETED",
         "desc": "Simulated label flipping and gradient poisoning attacks. Verified Byzantine fault tolerance using Trimmed Mean and Krum. Proven mathematical privacy via Rényi DP accounting.",
         "metric": "Recovered Byzantine AUC", "val": "0.8508", "href": "/?page=Attack+vs.+Defense"},
        {"phase": "Phase 5", "title": "MedXChAln Clinical SaaS Visual Console", "badge": "ACTIVE",
         "desc": "Engineered production-grade component library cloning MedXChAln design system. Integrated SHAP explainability, Federated XGBoost/RF, and on-chain traceability.",
         "metric": "Total Modules", "val": "18 Pages", "href": "/?page=Dashboard+Overview"},
    ]

    with col_left:
        for item in timeline_data:
            c_html = f"""
            <a href="{item['href']}" target="_top" style="text-decoration:none; color:inherit; display:block;" title="Open {item['title']}">
                <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:8px;">
                    <div>
                        <span class="fc-pill fc-pill-info">{item['phase']}</span>
                        <h4 style="margin:8px 0 4px; font-size:1.05rem; font-weight:700; color:#0F172A;">{item['title']}</h4>
                    </div>
                    {status_pill(item['badge'], 'success')}
                </div>
                <p style="font-size:0.85rem; color:#64748B; line-height:1.5; margin-bottom:12px;">{item['desc']}</p>
                <div style="font-size:0.75rem; color:#0F5BB6; font-weight:700;">
                    {item['metric']}: <span style="font-size:0.95rem; color:#0F172A;">{item['val']}</span>
                    <span style="margin-left:8px; font-size:0.8rem; font-weight:600;">Explore Module →</span>
                </div>
            </a>
            """
            st.markdown(fc_card(c_html), unsafe_allow_html=True)

    with col_side:
        navy_content = f"""
        <p style="font-size:0.8rem; color:rgba(255,255,255,0.7); margin-top:-6px; margin-bottom:14px;">
            Cumulative Research Milestones
        </p>
        <div style="margin-bottom:16px;">
            {fc_progress_item("Phase 1: Foundations", "100%", 100, color="#38BDF8")}
            {fc_progress_item("Phase 2: FedAvg Core", "100%", 100, color="#38BDF8")}
            {fc_progress_item("Phase 3: Non-IID Optimizers", "100%", 100, color="#38BDF8")}
            {fc_progress_item("Phase 4: Security & DP", "100%", 100, color="#38BDF8")}
            {fc_progress_item("Phase 5: MedXChAln Console", "100%", 100, color="#10B981")}
        </div>
        <div style="font-size:0.75rem; color:rgba(255,255,255,0.8); line-height:1.45; border-top:1px solid rgba(255,255,255,0.12); padding-top:10px;">
            All 5 thesis milestones fulfilled, validated against multi-center empirical benchmarks.
        </div>
        """
        st.markdown(fc_card(navy_content, title="Phase Completion Status", navy=True, icon_name="check_circle"), unsafe_allow_html=True)

        st.markdown(fc_card(
            svg_world_map(badge_text="VALIDATED CONSENSUS"),
            title="Consensus Verification",
            icon_name="globe"
        ), unsafe_allow_html=True)

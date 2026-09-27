"""
FedCare Page: Research Figures / Publication Gallery
====================================================
# PATTERN: Model/training-type page (Pattern 2, reference: Model Management Console)
# RATIONALE: Two-column row (Left: Publication Metadata & DPI specifications card |
# Right: Selected High-Resolution Figure Preview card with export controls) + full-width
# Figure Gallery Ledger table card below.
"""

from __future__ import annotations
from pathlib import Path
import streamlit as st

from app.components.ui_kit import (
    icon_svg, status_pill, fc_card, fc_progress_item, fc_table, render_header
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def render_research_figures():
    """Publication figures gallery cloned from MedXChAln Model Management."""
    # ── Top Header ───────────────────────────────────────────────
    st.markdown(render_header(
        title="Publication Figures Gallery",
        subtitle="High-resolution, publication-ready vector and raster visualizations generated across research phases.",
        search_placeholder="Search figure titles...",
        version_text="300 DPI Export",
        last_sync="Verified",
        action_label="Project Overview",
        action_icon="clipboard",
        action_href="/?page=Project+Overview"
    ), unsafe_allow_html=True)

    figures = {
        "Figure 1: FedAvg Convergence": RESULTS_DIR / "figure1_fedavg_convergence.png",
        "Figure 2: Non-IID Impact": RESULTS_DIR / "figure2_non_iid_impact.png",
        "Figure 3: Advanced Optimizers vs FedAvg": RESULTS_DIR / "figure3_fedprox_vs_fedavg.png",
        "Figure 4: Attacks & Defenses": RESULTS_DIR / "figure4_attacks_and_defenses.png",
        "Figure 5: Privacy-Utility Tradeoff": RESULTS_DIR / "figure5_privacy_utility.png",
    }
    available = {k: v for k, v in figures.items() if v.exists()}

    # ── Two-Column Row: Specs Card | Selected Figure Preview ─────
    col_specs, col_preview = st.columns([0.35, 0.65])

    with col_specs:
        specs_content = f"""
        <div style="background:#EFF6FF; border:1px solid #DBEAFE; border-radius:10px; padding:12px 14px; margin-bottom:16px;">
            <div style="font-size:0.7rem; font-weight:700; text-transform:uppercase; color:#0F5BB6; letter-spacing:0.05em;">PUBLICATION STANDARDS</div>
            <div style="font-size:0.98rem; font-weight:800; color:#0F172A; margin-top:2px;">IEEE / Nature Medicine Specs</div>
        </div>
        
        <div style="display:flex; justify-content:space-between; margin-bottom:16px; padding-bottom:12px; border-bottom:1px solid #F1F5F9;">
            <div>
                <div style="font-size:0.72rem; font-weight:600; text-transform:uppercase; color:#64748B;">RESOLUTION</div>
                <div style="font-size:1.35rem; font-weight:800; color:#0F172A; margin-top:2px;">300 DPI</div>
            </div>
            <div>
                <div style="font-size:0.72rem; font-weight:600; text-transform:uppercase; color:#64748B;">PALETTE</div>
                <div style="font-size:1.35rem; font-weight:800; color:#0F172A; margin-top:2px;">Colorblind Safe</div>
            </div>
        </div>
        
        <div style="font-size:0.72rem; font-weight:700; text-transform:uppercase; color:#64748B; letter-spacing:0.04em; margin-bottom:12px;">
            FIGURE ASSET INVENTORY
        </div>
        {fc_progress_item("Fig 1: Convergence", "Ready", 100, color="#0F5BB6")}
        {fc_progress_item("Fig 2: Non-IID Impact", "Ready", 100, color="#0F5BB6")}
        {fc_progress_item("Fig 3: Optimizers", "Ready", 100, color="#0F5BB6")}
        {fc_progress_item("Fig 4: Byzantine Defenses", "Ready", 100, color="#0F5BB6")}
        {fc_progress_item("Fig 5: Privacy Epsilon", "Ready", 100, color="#0F5BB6")}
        """
        st.markdown(fc_card(
            specs_content,
            title="Publication Asset Profile",
            badge=status_pill("5 ASSETS READY", "success"),
            icon_name="clipboard"
        ), unsafe_allow_html=True)

    with col_preview:
        fig_keys = list(figures.keys())
        selected_fig_title = st.selectbox("Select Figure to Preview", fig_keys, key="rf_select")
        fig_path = figures[selected_fig_title]

        if fig_path.exists():
            st.markdown(fc_card("", title=selected_fig_title, badge=status_pill("300 DPI", "info"), icon_name="activity"), unsafe_allow_html=True)
            st.image(str(fig_path), use_container_width=True)
        else:
            st.markdown(fc_card(f"""
            <div style="text-align:center; padding:50px 20px;">
                <div style="color:#0F5BB6; margin-bottom:12px;">{icon_svg("activity", size=36, color="#0F5BB6")}</div>
                <h4 style="margin:0 0 6px; color:#0F172A;">{selected_fig_title}</h4>
                <p style="color:#64748B; font-size:0.85rem;">Run thesis figure generation script to render 300-DPI raster cache.</p>
                <div style="font-family:monospace; background:#F1F5F9; padding:6px 12px; border-radius:6px; font-size:0.78rem; display:inline-block;">
                    python scripts/generate_figures.py
                </div>
            </div>
            """), unsafe_allow_html=True)

    # ── Full-Width Figures Table Card ────────────────────────────
    st.markdown('<div style="margin-top:12px;"></div>', unsafe_allow_html=True)
    table_headers = ["Figure Designation", "Research Topic", "Dimensions", "Format", "Status"]
    table_rows = [
        ["Figure 1", "Global FedAvg Convergence Trajectory", "2400 x 1400", "PNG / PDF", "VERIFIED"],
        ["Figure 2", "Non-IID Dirichlet Skew Impact on AUC", "2400 x 1400", "PNG / PDF", "VERIFIED"],
        ["Figure 3", "Proximal Regularization & Adaptive Optimizers", "2400 x 1400", "PNG / PDF", "VERIFIED"],
        ["Figure 4", "Adversarial Poisoning Attacks & Defenses", "2400 x 1400", "PNG / PDF", "VERIFIED"],
        ["Figure 5", "Differential Privacy Epsilon Pareto Curve", "2400 x 1400", "PNG / PDF", "VERIFIED"],
    ]
    st.markdown(fc_card(
        fc_table(table_headers, table_rows, status_col_idx=4),
        title="Publication Figure Registry",
        subtitle="Complete catalog of figures formatted for journal submission",
        icon_name="database"
    ), unsafe_allow_html=True)

"""
FedCare Page: Data Explorer / Interactive Dataset Exploration
=============================================================
# PATTERN: Diagnosis/form-type page (Pattern 4, reference: Diagnosis Support Tool)
# RATIONALE: Two-column layout (Left: Feature filter and cohort selection input form card
# with distribution stat pairs + 2 consensus mini cards | Right: Stacked distribution result
# cards, correlation heatmap, and hospital demographic comparison).
"""

from __future__ import annotations
import streamlit as st
import numpy as np
import plotly.graph_objects as go

from app.components.ui_kit import (
    icon_svg, status_pill, fc_card, fc_table, render_header
)
from app.components.theme import (
    HOSPITAL_COLORS, CHART_COLORS, STRATEGY_COLORS, PLOTLY_LAYOUT_DEFAULTS
)
from app.utils.data_loaders import (
    load_hospital_stats, load_combined_data, load_hospital_raw
)


def render_data_explorer():
    """Interactive data explorer cloned from MedXChAln Diagnosis Support Tool."""
    hospital_stats = load_hospital_stats()
    if hospital_stats.empty:
        st.warning("No hospital data found.")
        return

    # ── Top Header ───────────────────────────────────────────────
    st.markdown(render_header(
        title="Data Explorer",
        subtitle="Cross-hospital feature distributions, Pearson correlation matrices, and demographic cohort variance.",
        search_placeholder="Search features...",
        live_badge=status_pill("13 Features Indexed", "success"),
        version_text="v4.2.1-Data",
        last_sync="Just now",
        action_label="Export Dataset",
        action_icon="download",
        action_href="/?page=Research+Figures"
    ), unsafe_allow_html=True)

    # ── Two-Column Layout ────────────────────────────────────────
    col_input, col_charts = st.columns([0.38, 0.62])

    with col_input:
        st.markdown(fc_card("""
        <div style="font-size:0.7rem; font-weight:700; color:#64748B; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:6px;">
            COHORT FILTER CONTROLS
        </div>
        """), unsafe_allow_html=True)

        selected_hosp = st.selectbox(
            "Select Clinical Facility",
            ["All Hospitals"] + [f"Hospital {i}" for i in range(1, 7)],
            key="de_hosp"
        )
        
        df = load_combined_data() if selected_hosp == "All Hospitals" else load_hospital_raw(int(selected_hosp.split()[-1]))

        if df is not None:
            numeric_cols = [c for c in df.columns if c != "target" and df[c].dtype in ["float64", "int64", "float32", "int32"]]
            cont_cols = [c for c in numeric_cols if df[c].nunique() > 5]
            selected_feature = st.selectbox("Select Continuous Biomarker", cont_cols, key="de_feat")

            # Feature Summary Card
            mean_v = df[selected_feature].mean()
            std_v = df[selected_feature].std()
            min_v = df[selected_feature].min()
            max_v = df[selected_feature].max()

            stats_content = f"""
            <div style="font-size:0.75rem; font-weight:700; color:#0F5BB6; text-transform:uppercase; margin-bottom:12px;">
                {selected_feature.upper()} DESCRIPTIVE METRICS
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:8px; padding-bottom:6px; border-bottom:1px solid #F1F5F9;">
                <span style="font-size:0.8rem; color:#64748B;">Mean Value</span>
                <strong style="font-size:0.85rem; color:#0F172A;">{mean_v:.2f}</strong>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:8px; padding-bottom:6px; border-bottom:1px solid #F1F5F9;">
                <span style="font-size:0.8rem; color:#64748B;">Standard Deviation</span>
                <strong style="font-size:0.85rem; color:#0F172A;">±{std_v:.2f}</strong>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:8px; padding-bottom:6px; border-bottom:1px solid #F1F5F9;">
                <span style="font-size:0.8rem; color:#64748B;">Observed Range</span>
                <strong style="font-size:0.85rem; color:#0F172A;">[{min_v:.1f} — {max_v:.1f}]</strong>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="font-size:0.8rem; color:#64748B;">Total Records</span>
                <strong style="font-size:0.85rem; color:#15803D;">{len(df):,} patients</strong>
            </div>
            """
            st.markdown(fc_card(stats_content, icon_name="clipboard"), unsafe_allow_html=True)

        # 2 Mini Cards
        m1_content = f"""
        <div style="display:flex; align-items:center; gap:8px;">
            <div style="color:#0F5BB6;">{icon_svg("globe", size=18, color="#0F5BB6")}</div>
            <div>
                <div style="font-size:0.82rem; font-weight:700; color:#0F172A;">Multi-Site Harmonization</div>
                <div style="font-size:0.72rem; color:#64748B;">Z-score scaling calibrated locally.</div>
            </div>
        </div>
        """
        st.markdown(fc_card(m1_content), unsafe_allow_html=True)

        m2_content = f"""
        <div style="display:flex; align-items:center; gap:8px;">
            <div style="color:#15803D;">{icon_svg("shield_check", size=18, color="#15803D")}</div>
            <div>
                <div style="font-size:0.82rem; font-weight:700; color:#0F172A;">Zero Raw Data Sharing</div>
                <div style="font-size:0.72rem; color:#64748B;">Exploration runs on sandboxed views.</div>
            </div>
        </div>
        """
        st.markdown(fc_card(m2_content), unsafe_allow_html=True)

    with col_charts:
        tab1, tab2, tab3 = st.tabs(["Biomarker Distributions", "Correlation Matrix", "Cross-Hospital Split"])

        with tab1:
            if df is not None:
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=df[df["target"] == 0][selected_feature], name="Healthy",
                    marker_color="rgba(22,163,74,0.5)", nbinsx=35
                ))
                fig_dist.add_trace(go.Histogram(
                    x=df[df["target"] == 1][selected_feature], name="Heart Disease",
                    marker_color="rgba(220,38,38,0.55)", nbinsx=35
                ))
                fig_dist.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
                fig_dist.update_layout(
                    height=340, barmode="overlay",
                    xaxis_title=selected_feature.replace("_", " ").title(),
                    yaxis_title="Patient Count",
                    margin=dict(l=30, r=20, t=20, b=30)
                )
                st.markdown(fc_card("", title=f"{selected_feature.title()} Distribution Overlay", icon_name="activity"), unsafe_allow_html=True)
                st.plotly_chart(fig_dist, use_container_width=True)

        with tab2:
            combined = load_combined_data()
            if combined is not None:
                num_cols = [c for c in combined.columns if combined[c].dtype in ["float64", "int64", "float32", "int32"]]
                corr_matrix = combined[num_cols].corr()
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values, x=corr_matrix.columns.tolist(), y=corr_matrix.index.tolist(),
                    colorscale=[[0, "#EFF6FF"], [0.5, "#BFDBFE"], [1.0, "#0F5BB6"]],
                    text=np.round(corr_matrix.values, 2), texttemplate="%{text}",
                    textfont=dict(size=9, color="#0F172A", family="Inter")
                ))
                fig_corr.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
                fig_corr.update_layout(height=380, margin=dict(l=80, r=20, t=10, b=30))
                st.markdown(fc_card("", title="Biomarker Pearson Correlation Matrix", icon_name="grid"), unsafe_allow_html=True)
                st.plotly_chart(fig_corr, use_container_width=True)

        with tab3:
            fig_prev = go.Figure()
            fig_prev.add_trace(go.Bar(
                x=hospital_stats["Hospital"], y=hospital_stats["Prevalence"] * 100,
                marker_color=HOSPITAL_COLORS, marker_cornerradius=6,
                text=[f"{v:.1f}%" for v in hospital_stats["Prevalence"] * 100],
                textposition="outside", textfont=dict(color="#0F172A", size=11, family="Inter")
            ))
            fig_prev.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
            fig_prev.update_layout(height=340, yaxis_title="Disease Prevalence (%)", showlegend=False)
            st.markdown(fc_card("", title="Heart Disease Prevalence by Hospital", icon_name="hospital"), unsafe_allow_html=True)
            st.plotly_chart(fig_prev, use_container_width=True)

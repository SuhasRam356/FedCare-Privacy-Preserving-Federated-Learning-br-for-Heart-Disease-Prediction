"""
FedCare Page: Anamnesis Record / Clinical Patient Intake
========================================================
# PATTERN: Diagnosis/form-type page (Pattern 4, reference: Diagnosis Support Tool / Doctor Console)
# RATIONALE: Two-column layout (Left: Medical history intake form with grouped patient symptoms,
# vitals, and Encrypted Local Processing button + 2 consensus mini cards | Right: Stacked
# AI diagnostic inference result cards, next suggested lab panel, and model interpretability).
"""

from __future__ import annotations
import streamlit as st

from app.components.ui_kit import (
    icon_svg, status_pill, fc_card, fc_progress_item, fc_fab, render_header, clean_html
)



def render_anamnesis_record():
    """Patient anamnesis intake console cloned from MedXChAln Diagnosis Support Tool."""
    # ── Top Header ───────────────────────────────────────────────
    st.markdown(render_header(
        title="Anamnesis Record",
        subtitle="Patient medical history intake, structured EHR entry, and automated differential diagnosis support.",
        search_placeholder="Search patient record or MRN...",
        live_badge=status_pill("CONFIDENTIAL EHR", "info"),
        version_text="v4.2.1-Clinician",
        last_sync="Active",
        action_label="New Intake",
        action_icon="user"
    ), unsafe_allow_html=True)

    # ── Two-Column Layout ────────────────────────────────────────
    col_form, col_results = st.columns([0.58, 0.42])

    with col_form:
        st.markdown(clean_html("""
        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:12px;">
            <div>
                <h3 style="margin:0; font-size:1.2rem; font-weight:800; color:#0F172A;">Patient Clinical Intake</h3>
                <p style="margin:2px 0 0; font-size:0.8rem; color:#64748B;">Records encrypted locally prior to federated model evaluation.</p>
            </div>
            <div style="color:#0F5BB6;">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#0F5BB6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>
                </svg>
            </div>
        </div>
        """), unsafe_allow_html=True)

        with st.form("anamnesis_form"):
            p1, p2, p3 = st.columns(3)
            with p1:
                mrn = st.text_input("PATIENT MRN", value="PT-89412")
            with p2:
                age = st.number_input("AGE (YRS)", 18, 100, 61)
            with p3:
                gender = st.selectbox("BIOLOGICAL SEX", ["Male", "Female"])

            chief_complaint = st.text_area(
                "CHIEF COMPLAINT & SYMPTOMS",
                value="Recurrent substernal chest pressure upon moderate exertion, radiating to the left shoulder. Associated with dyspnea on walking uphill.",
                height=80
            )

            v1, v2, v3 = st.columns(3)
            with v1:
                bp = st.text_input("BP (MMHG)", value="138/86")
            with v2:
                hr = st.number_input("HEART RATE (BPM)", 50, 200, 78)
            with v3:
                bmi = st.number_input("BMI (KG/M²)", 15.0, 50.0, 29.1, 0.1)

            h1, h2, h3 = st.columns(3)
            with h1:
                smoker = st.selectbox("SMOKING HISTORY", ["Current", "Former", "Never"])
            with h2:
                diabetes = st.selectbox("DIABETES (TYPE 2)", ["Present", "Absent", "Prediabetic"])
            with h3:
                family_cad = st.selectbox("FAMILY HISTORY OF CAD", ["Confirmed", "Unconfirmed", "Negative"])

            med_notes = st.text_input(
                "CURRENT MEDICATIONS & CONTRAINDICATIONS",
                value="Atorvastatin 20mg q.d., Metoprolol 50mg b.i.d."
            )

            submit_btn = st.form_submit_button("⚡ Evaluate Against Global Model", use_container_width=True)

        # 2 Mini Cards
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(fc_card(f"""
            <div style="display:flex; align-items:center; gap:8px;">
                <div style="color:#0F5BB6;">{icon_svg("globe", size=18, color="#0F5BB6")}</div>
                <div>
                    <div style="font-size:0.82rem; font-weight:700; color:#0F172A;">Consensus Benchmarking</div>
                    <div style="font-size:0.72rem; color:#64748B;">Cross-checked against 2,990 clinical records.</div>
                </div>
            </div>
            """), unsafe_allow_html=True)

        with m2:
            st.markdown(fc_card(f"""
            <div style="display:flex; align-items:center; gap:8px;">
                <div style="color:#15803D;">{icon_svg("shield_check", size=18, color="#15803D")}</div>
                <div>
                    <div style="font-size:0.82rem; font-weight:700; color:#0F172A;">HIPAA Tier 3 Safe Harbor</div>
                    <div style="font-size:0.72rem; color:#64748B;">Direct identifiers stripped in memory.</div>
                </div>
            </div>
            """), unsafe_allow_html=True)

    with col_results:
        st.markdown(clean_html("""
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:14px; border-left:4px solid #0F5BB6; padding-left:10px;">
            <h3 style="margin:0; font-size:1.15rem; font-weight:800; color:#0F172A;">Diagnostic Consensus Inferences</h3>
        </div>
        """), unsafe_allow_html=True)

        # Result card 1
        c1_html = f"""
        <div style="display:flex; justify-content:space-between; align-items:baseline;">
            <div>
                <div style="font-size:1.05rem; font-weight:800; color:#0F172A;">Coronary Artery Disease</div>
                <div style="font-size:0.7rem; font-weight:700; color:#94A3B8; letter-spacing:0.04em;">PREDICTED PROBABILITY</div>
            </div>
            <div style="font-size:1.85rem; font-weight:800; color:#0F5BB6; font-family:'Inter', sans-serif;">
                88%
            </div>
        </div>
        <div style="width:100%; height:7px; background:#E2E8F0; border-radius:999px; overflow:hidden; margin:10px 0 8px;">
            <div style="width:88%; height:100%; background:#0F5BB6; border-radius:999px;"></div>
        </div>
        <div style="font-size:0.75rem; color:#15803D; font-weight:600; display:flex; align-items:center; gap:5px;">
            <span>{icon_svg("node_tree", size=13, color="#15803D")}</span>
            <span>Matched to high-risk exertional angina cluster</span>
        </div>
        """
        st.markdown(fc_card(c1_html), unsafe_allow_html=True)

        # Result card 2
        c2_html = f"""
        <div style="display:flex; justify-content:space-between; align-items:baseline;">
            <div>
                <div style="font-size:1.05rem; font-weight:800; color:#0F172A;">Cardiac Arrhythmia Potential</div>
                <div style="font-size:0.7rem; font-weight:700; color:#94A3B8; letter-spacing:0.04em;">SECONDARY RISK</div>
            </div>
            <div style="font-size:1.85rem; font-weight:800; color:#64748B; font-family:'Inter', sans-serif;">
                42%
            </div>
        </div>
        <div style="width:100%; height:7px; background:#E2E8F0; border-radius:999px; overflow:hidden; margin:10px 0 8px;">
            <div style="width:42%; height:100%; background:#94A3B8; border-radius:999px;"></div>
        </div>
        <div style="font-size:0.75rem; color:#64748B; font-weight:500;">
            Recommend 12-lead resting and stress ECG
        </div>
        """
        st.markdown(fc_card(c2_html), unsafe_allow_html=True)

        # Navy Action Card: Recommended Clinical Pathway
        navy_pathway_html = f"""
        <a href="/?page=Diagnosis+Support" target="_top" style="text-decoration:none; color:inherit; display:block;" title="Run Differential Diagnosis Inference">
            <div style="display:flex; align-items:center; justify-content:space-between; cursor:pointer;">
                <div style="display:flex; align-items:center; gap:10px;">
                    <div style="color:#60A5FA;">
                        {icon_svg("stethoscope", size=20, color="#60A5FA")}
                    </div>
                    <div>
                        <div style="font-size:0.72rem; color:rgba(255,255,255,0.7); font-weight:600; text-transform:uppercase;">Suggested Next Step</div>
                        <div style="font-size:0.85rem; font-weight:700; color:#FFFFFF;">Schedule Cardiac Catheterization / CTA →</div>
                    </div>
                </div>
                <div style="color:rgba(255,255,255,0.6);">
                    {icon_svg("chevron_right", size=16, color="#FFFFFF")}
                </div>
            </div>
        </a>
        """
        st.markdown(fc_card(navy_pathway_html, navy=True), unsafe_allow_html=True)

        # Bottom SHAP Interpretability
        st.markdown(fc_card(f"""
        <div style="font-size:0.7rem; font-weight:700; color:#64748B; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:12px;">
            PATIENT BIOMARKER INFLUENCE BREAKDOWN
        </div>
        <div style="margin-bottom:10px;">
            {fc_progress_item("Exertional Chest Pain (Typical)", "+0.42 log-odds", 90, color="#EF4444", href="/?page=Feature+Importance")}
            {fc_progress_item("Elevated Resting BP (138 mmHg)", "+0.28 log-odds", 70, color="#F59E0B", href="/?page=Feature+Importance")}
            {fc_progress_item("Active Smoking Status", "+0.22 log-odds", 60, color="#F59E0B", href="/?page=Feature+Importance")}
            {fc_progress_item("Heart Rate Reserve (78 bpm)", "-0.14 protective", 40, color="#10B981", href="/?page=Feature+Importance")}
        </div>
        <div style="text-align:center; font-size:0.7rem; color:#94A3B8;">
            SHAP LOCAL EXPLANATION CONSISTENT WITH GLOBAL MODEL
        </div>
        """), unsafe_allow_html=True)

    # FAB Button
    st.markdown(fc_fab(icon_name="printer", title="Print Patient Summary", href="javascript:window.print()"), unsafe_allow_html=True)

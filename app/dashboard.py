"""
FedCare Interactive Dashboard — MedXChAln Clinical SaaS Redesign
=================================================================
Privacy-Preserving Federated Learning for Heart Disease Prediction.
Cloned to the MedXChAln Clinical SaaS reference design system.

Author: FedCare Team
"""

from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st

# ── Path Setup ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Streamlit Page Configuration ──────────────────────────────────────
st.set_page_config(
    page_title="MedXChAIn | FedCare — Federated Clinical Console",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "FedCare - Privacy-Preserving Federated Learning for Heart Disease Prediction (MedXChAln System)"
    }
)

# ── UI Kit & Theme Imports ────────────────────────────────────────────
from app.components.theme import inject_clinical_theme
from app.components.ui_kit import (
    render_sidebar_header, render_sidebar_footer, icon_svg
)
from app.utils.data_loaders import load_hospital_stats

# ── Modular Page Renderers (18 Pages) ─────────────────────────────────
from app.pages import (
    render_command_center,
    render_anamnesis_record,
    render_risk_calculator,
    render_training_console,
    render_network_topology,
    render_secure_aggregation,
    render_project_overview,
    render_live_stream,
    render_communication_cost,
    render_attack_defense,
    render_privacy_utility,
    render_non_iid_analysis,
    render_feature_importance,
    render_model_comparison,
    render_data_explorer,
    render_hospital_deep_dive,
    render_experiment_timeline,
    render_research_figures,
)


# ══════════════════════════════════════════════════════════════════════
#                         MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════

CANONICAL_PAGES = {
    "dashboard overview": "Dashboard Overview",
    "overview": "Dashboard Overview",
    "dashboard": "Dashboard Overview",
    "command center": "Dashboard Overview",
    "anamnesis record": "Anamnesis Record",
    "anamnesis": "Anamnesis Record",
    "doctor console": "Anamnesis Record",
    "patient": "Anamnesis Record",
    "diagnosis support": "Diagnosis Support",
    "risk calculator": "Diagnosis Support",
    "model management": "Model Management",
    "training console": "Model Management",
    "hospital management": "Hospital Management",
    "network topology": "Hospital Management",
    "security & traceability": "Security & Traceability",
    "security and traceability": "Security & Traceability",
    "secure aggregation": "Security & Traceability",
    "project overview": "Project Overview",
    "live training stream": "Live Training Stream",
    "communication cost": "Communication Cost",
    "attack vs. defense": "Attack vs. Defense",
    "attack vs defense": "Attack vs. Defense",
    "privacy-utility": "Privacy-Utility",
    "privacy utility": "Privacy-Utility",
    "non-iid analysis": "Non-IID Analysis",
    "non iid analysis": "Non-IID Analysis",
    "feature importance": "Feature Importance",
    "model comparison": "Model Comparison",
    "data explorer": "Data Explorer",
    "hospital deep dive": "Hospital Deep Dive",
    "experiment timeline": "Experiment Timeline",
    "research figures": "Research Figures",
}

PAGE_KEYWORDS = [
    (["hospital deep dive", "deep dive", "deepdive", "mayo", "cleveland", "hopkins", "charite", "zurich"], "Hospital Deep Dive"),
    (["project overview", "about", "guide", "manual", "documentation"], "Project Overview"),
    (["dash", "home", "command"], "Dashboard Overview"),
    (["anamnesis", "intake", "doctor", "ehr", "mrn", "history"], "Anamnesis Record"),
    (["diag", "risk", "inference", "calculat", "symptom", "suggestion"], "Diagnosis Support"),
    (["model", "train", "architect", "epoch", "dense", "param", "aggregat"], "Model Management"),
    (["hospital management", "hosp", "node", "facilit", "network", "particip", "topolog"], "Hospital Management"),
    (["secur", "trace", "ledger", "block", "chain", "ipfs", "audit", "sha", "proof"], "Security & Traceability"),
    (["live", "stream", "event", "realtime", "telemetry"], "Live Training Stream"),
    (["comm", "cost", "bandwidth", "traffic", "network cost"], "Communication Cost"),
    (["attack", "defense", "poison", "byzantine", "adversar"], "Attack vs. Defense"),
    (["priv", "utility", "dp", "epsilon", "differential", "noise"], "Privacy-Utility"),
    (["non-iid", "noniid", "dirichlet", "heterogen", "skew"], "Non-IID Analysis"),
    (["feature", "shap", "importan", "explain", "interpret"], "Feature Importance"),
    (["compare", "comparison", "xgboost", "random forest", "ensemble", "benchmark"], "Model Comparison"),
    (["data", "explorer", "dataset", "biomarker", "cohort"], "Data Explorer"),
    (["timeline", "experiment", "milestone"], "Experiment Timeline"),
    (["figure", "chart", "publication", "paper", "export"], "Research Figures"),
]

def search_to_page(query: str) -> str:
    q = (query or "").strip().lower()
    if q in CANONICAL_PAGES:
        return CANONICAL_PAGES[q]
    for keywords, page_name in PAGE_KEYWORDS:
        for kw in keywords:
            if kw in q:
                return page_name
    return "Dashboard Overview"

def normalize_page_name(raw: str) -> str:
    r = (raw or "").strip().lower().replace("_", " ").replace("-", " ")
    if r in CANONICAL_PAGES:
        return CANONICAL_PAGES[r]
    # Check exact match with canonical keys after stripping punctuation
    r_clean = r.replace("&", "and").strip()
    if r_clean in CANONICAL_PAGES:
        return CANONICAL_PAGES[r_clean]
    for keywords, page_name in PAGE_KEYWORDS:
        if r == page_name.lower():
            return page_name
        for kw in keywords:
            if kw == r or kw in r:
                return page_name
    return ""

RESEARCH_PAGES = {
    "Project Overview",
    "Live Training Stream",
    "Communication Cost",
    "Attack vs. Defense",
    "Privacy-Utility",
    "Non-IID Analysis",
    "Feature Importance",
    "Model Comparison",
    "Data Explorer",
    "Hospital Deep Dive",
    "Experiment Timeline",
    "Research Figures",
}


def switch_page(new_page: str):
    """Switch active page, synchronize query parameter, and rerun cleanly."""
    st.session_state.active_page = new_page
    st.session_state["_last_handled_qp"] = new_page
    try:
        st.query_params["page"] = new_page
    except Exception:
        pass
    st.rerun()


def main():
    # Session state for page navigation
    if 'active_page' not in st.session_state:
        st.session_state.active_page = "Project Overview"

    # ── Interactive Query Parameter Routing & Action Triggers ───────
    qp = st.query_params
    if "refresh" in qp:
        st.cache_data.clear()
        try:
            del st.query_params["refresh"]
        except Exception:
            pass

    if "search" in qp:
        search_term = qp.get("search", "")
        target = search_to_page(search_term)
        if target:
            st.session_state.active_page = target
            st.session_state["_last_handled_qp"] = target
            try:
                st.query_params["page"] = target
            except Exception:
                pass
        try:
            del st.query_params["search"]
        except Exception:
            pass

    if "page" in qp:
        page_req = qp.get("page", "")
        # Only adopt query param if it has changed from what was already handled
        if st.session_state.get("_last_handled_qp") != page_req:
            matched = normalize_page_name(page_req)
            if matched:
                st.session_state.active_page = matched
            st.session_state["_last_handled_qp"] = page_req
    else:
        if "_last_handled_qp" not in st.session_state:
            st.session_state["_last_handled_qp"] = st.session_state.active_page

    # Map current active page to its corresponding button key for CSS conditional styling
    page_to_key = {
        "Dashboard Overview": "nav_dash",
        "Overview": "nav_dash",
        "Command Center": "nav_dash",
        "Anamnesis Record": "nav_anamnesis",
        "Diagnosis Support": "nav_risk",
        "Risk Calculator": "nav_risk",
        "Model Management": "nav_train",
        "Training Console": "nav_train",
        "Hospital Management": "nav_net",
        "Network Topology": "nav_net",
        "Security & Traceability": "nav_secagg",
        "Secure Aggregation": "nav_secagg",
        "Project Overview": "nav_proj",
        "Live Training Stream": "nav_live",
        "Communication Cost": "nav_comm",
        "Attack vs. Defense": "nav_attack",
        "Privacy-Utility": "nav_priv",
        "Non-IID Analysis": "nav_noniid",
        "Feature Importance": "nav_shap",
        "Model Comparison": "nav_model",
        "Data Explorer": "nav_data",
        "Hospital Deep Dive": "nav_hosp",
        "Experiment Timeline": "nav_timeline",
        "Research Figures": "nav_figs",
    }
    
    current_key = page_to_key.get(st.session_state.active_page, "nav_dash")
    inject_clinical_theme(active_page_key=current_key)

    hospital_stats = load_hospital_stats()
    n_nodes = len(hospital_stats) if not hospital_stats.empty else 6

    # ══════════════════════════════════════════════════════════════════
    #                     COMPONENT 1 — SIDEBAR
    # ══════════════════════════════════════════════════════════════════
    with st.sidebar:
        # 1. Logo Row & 2. Identity Block
        st.markdown(render_sidebar_header(), unsafe_allow_html=True)

        # 3. Federated Learning Navigation Items
        st.markdown('<div class="fc-sidebar-section-title">FEDERATED LEARNING</div>', unsafe_allow_html=True)
        
        if st.button("📋 Project Overview", key="nav_proj", use_container_width=True):
            switch_page("Project Overview")

        if st.button("📡 Live Training Stream", key="nav_live", use_container_width=True):
            switch_page("Live Training Stream")

        if st.button("📶 Communication Cost", key="nav_comm", use_container_width=True):
            switch_page("Communication Cost")

        if st.button("🛡️ Attack vs. Defense", key="nav_attack", use_container_width=True):
            switch_page("Attack vs. Defense")

        if st.button("🔐 Privacy-Utility", key="nav_priv", use_container_width=True):
            switch_page("Privacy-Utility")

        if st.button("📊 Non-IID Analysis", key="nav_noniid", use_container_width=True):
            switch_page("Non-IID Analysis")

        if st.button("🧠 Feature Importance", key="nav_shap", use_container_width=True):
            switch_page("Feature Importance")

        if st.button("⚖️ Model Comparison", key="nav_model", use_container_width=True):
            switch_page("Model Comparison")

        if st.button("🔬 Data Explorer", key="nav_data", use_container_width=True):
            switch_page("Data Explorer")

        if st.button("🌐 Node Deep Dive", key="nav_hosp", use_container_width=True):
            switch_page("Hospital Deep Dive")

        if st.button("🗺️ Experiment Timeline", key="nav_timeline", use_container_width=True):
            switch_page("Experiment Timeline")

        if st.button("🎨 Research Figures", key="nav_figs", use_container_width=True):
            switch_page("Research Figures")

        # 4. Spacer, 5. Status Card, 6. Settings & Support
        st.markdown(render_sidebar_footer(nodes_count=n_nodes, sync_status="Active"), unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    #                     PAGE ROUTING DISPATCHER
    # ══════════════════════════════════════════════════════════════════
    page = st.session_state.active_page

    if page in ["Dashboard Overview", "Command Center", "Overview"]:
        render_command_center()
    elif page == "Anamnesis Record":
        render_anamnesis_record()
    elif page in ["Diagnosis Support", "Risk Calculator"]:
        render_risk_calculator()
    elif page in ["Model Management", "Training Console"]:
        render_training_console()
    elif page in ["Hospital Management", "Network Topology"]:
        render_network_topology()
    elif page in ["Security & Traceability", "Secure Aggregation"]:
        render_secure_aggregation()
    elif page == "Project Overview":
        render_project_overview()
    elif page == "Live Training Stream":
        render_live_stream()
    elif page == "Communication Cost":
        render_communication_cost()
    elif page == "Attack vs. Defense":
        render_attack_defense()
    elif page == "Privacy-Utility":
        render_privacy_utility()
    elif page == "Non-IID Analysis":
        render_non_iid_analysis()
    elif page == "Feature Importance":
        render_feature_importance()
    elif page == "Model Comparison":
        render_model_comparison()
    elif page == "Data Explorer":
        render_data_explorer()
    elif page == "Hospital Deep Dive":
        render_hospital_deep_dive()
    elif page == "Experiment Timeline":
        render_experiment_timeline()
    elif page == "Research Figures":
        render_research_figures()
    else:
        render_command_center()


if __name__ == "__main__":
    main()

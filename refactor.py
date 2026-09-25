import re

with open("app/dashboard.py", "r") as f:
    content = f.read()

# 1. Replace Design System section (lines 55-98 approx)
content = re.sub(
    r"# ══════════════════════════════════════════════════════════════════════\n#                         DESIGN SYSTEM v3\n# ══════════════════════════════════════════════════════════════════════.*?PLOTLY_THEME = dict\([\s\S]*?yaxis=dict\(.*?\),\n\)",
    """# ══════════════════════════════════════════════════════════════════════
#                         IMPORTS & SETUP
# ══════════════════════════════════════════════════════════════════════

from app.components.theme import (
    inject_clinical_theme, render_page_header, 
    HOSPITAL_COLORS, STRATEGY_COLORS, CHART_COLORS, PLOTLY_LAYOUT_DEFAULTS
)

PLOTLY_THEME = PLOTLY_LAYOUT_DEFAULTS""",
    content, flags=re.DOTALL
)

# 2. Empty inject_css
content = re.sub(
    r"def inject_css\(\):[\s\S]*?    </style>\"\n    \"\"\", unsafe_allow_html=True\)",
    "",
    content
)

# 3. Replace Data Loaders section
content = re.sub(
    r"# ══════════════════════════════════════════════════════════════════════\n#                     CACHED DATA LOADERS\n# ══════════════════════════════════════════════════════════════════════[\s\S]*?return model, scaler",
    """from app.utils.data_loaders import (
    load_hospital_stats, load_hospital_raw, load_combined_data, load_fedavg_rounds,
    load_attack_defense_matrix, load_dp_sweep, load_non_iid_results, load_fedprox_results,
    load_comm_cost, load_attack_trajectories, load_global_model
)""",
    content, flags=re.DOTALL
)

# 4. Fix bandwidth math in render_communication_cost
content = re.sub(
    r'        ratio = row\["comm_ratio"\]\n        savings = abs\(\(1 - ratio\) \* 100\)\n        st.metric\("vs. Centralized", f"\{ratio:\.2f\}x", delta=f"\{savings:\.1f\}% savings"\)',
    """        ratio = row["comm_ratio"]
        overhead = (ratio - 1) * 100
        st.metric("vs. Centralized", f"{ratio:.2f}x", delta=f"-{overhead:.1f}% overhead", delta_color="inverse")""",
    content
)

content = re.sub(
    r'marker_color=\["#00d4ff", "#fb7185"\], marker_cornerradius=8,',
    r'marker_color=[CHART_COLORS["primary"], CHART_COLORS["danger"]], marker_cornerradius=8,',
    content
)

# 5. Fix main() function navigation
main_code = """def main():
    inject_clinical_theme()

    if 'active_page' not in st.session_state:
        st.session_state.active_page = "Project Overview"

    with st.sidebar:
        st.markdown("<h2 style='text-align:center; color:#3b82f6;'>FedCare</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color:#94a3b8; font-size:0.85rem; margin-top:-10px;'>Clinical Research Platform</p>", unsafe_allow_html=True)
        st.markdown('<hr style="margin: 10px 0;">', unsafe_allow_html=True)
        
        # Group 1: Overview
        st.markdown('**Overview**')
        if st.button("Project Overview", use_container_width=True): st.session_state.active_page = "Project Overview"
        if st.button("Dashboard Overview", use_container_width=True): st.session_state.active_page = "Dashboard Overview"
        
        st.markdown('**Training & Network**')
        if st.button("Network Topology", use_container_width=True): st.session_state.active_page = "Network Topology"
        if st.button("Training Console", use_container_width=True): st.session_state.active_page = "Training Console"
        if st.button("Communication Cost", use_container_width=True): st.session_state.active_page = "Communication Cost"
        
        st.markdown('**Security & Privacy**')
        if st.button("Secure Aggregation", use_container_width=True): st.session_state.active_page = "Secure Aggregation"
        if st.button("Attack vs. Defense", use_container_width=True): st.session_state.active_page = "Attack vs. Defense"
        if st.button("Privacy-Utility", use_container_width=True): st.session_state.active_page = "Privacy-Utility"
        if st.button("Non-IID Analysis", use_container_width=True): st.session_state.active_page = "Non-IID Analysis"
        
        st.markdown('**Tools & Insights**')
        if st.button("Risk Calculator", use_container_width=True): st.session_state.active_page = "Risk Calculator"
        if st.button("Feature Importance", use_container_width=True): st.session_state.active_page = "Feature Importance"
        if st.button("Model Comparison", use_container_width=True): st.session_state.active_page = "Model Comparison"
        if st.button("Data Explorer", use_container_width=True): st.session_state.active_page = "Data Explorer"
        if st.button("Hospital Deep Dive", use_container_width=True): st.session_state.active_page = "Hospital Deep Dive"
        if st.button("Experiment Timeline", use_container_width=True): st.session_state.active_page = "Experiment Timeline"
        if st.button("Research Figures", use_container_width=True): st.session_state.active_page = "Research Figures"
        
        st.markdown('<hr style="margin: 10px 0;">', unsafe_allow_html=True)
        with st.expander("Dataset Summary"):
            hospital_stats = load_hospital_stats()
            if not hospital_stats.empty:
                st.metric("Total Hospitals", f"{len(hospital_stats)}")
                st.metric("Total Patients", f"{hospital_stats['Samples'].sum():,}")
                st.metric("Overall Prevalence", f"{(hospital_stats['Positive'].sum() / hospital_stats['Samples'].sum()) * 100:.1f}%")

    page = st.session_state.active_page
    
    if page == "Project Overview":
        render_page_header("Project Overview", "Privacy-Preserving Federated Learning for Heart Disease Prediction")
        render_project_overview()
    elif page == "Dashboard Overview":
        render_page_header("Command Center", "High-level metrics and research phase progress.")
        render_command_center()
    elif page == "Network Topology":
        render_page_header("Network Topology", "Federated network architecture and hospital participants.")
        render_network_topology()
    elif page == "Training Console":
        render_page_header("Training Console", "Round-by-round convergence and inter-hospital equity.")
        render_training_console()
    elif page == "Communication Cost":
        render_page_header("Communication Cost", "Bandwidth analysis of federated vs centralized learning.")
        render_communication_cost()
    elif page == "Secure Aggregation":
        render_page_header("Secure Aggregation", "Cryptographic privacy via Secret Sharing and Homomorphic Encryption.")
        render_secure_aggregation()
    elif page == "Attack vs. Defense":
        render_page_header("Attack vs. Defense", "Evaluating Byzantine robustness against adversarial attacks.")
        render_attack_defense()
    elif page == "Privacy-Utility":
        render_page_header("Privacy-Utility Tradeoff", "Differential Privacy noise multiplier analysis.")
        render_privacy_utility()
    elif page == "Non-IID Analysis":
        render_page_header("Non-IID Analysis", "Impact of data heterogeneity across hospitals.")
        render_non_iid_analysis()
    elif page == "Risk Calculator":
        render_page_header("Risk Calculator", "Live clinical prediction using the global federated model.")
        render_risk_calculator()
    elif page == "Feature Importance":
        render_page_header("Feature Importance", "Global feature influence explained by SHAP.")
        render_feature_importance()
    elif page == "Model Comparison":
        render_page_header("Model Comparison", "MLP vs Federated XGBoost vs Federated Random Forest.")
        render_model_comparison()
    elif page == "Data Explorer":
        render_page_header("Data Explorer", "Interactive feature distributions and correlation matrix.")
        render_data_explorer()
    elif page == "Hospital Deep Dive":
        render_page_header("Hospital Deep Dive", "In-depth analysis of individual hospital cohorts.")
        render_hospital_deep_dive()
    elif page == "Experiment Timeline":
        render_page_header("Experiment Timeline", "Step-by-step progress through the 5 research phases.")
        render_experiment_timeline()
    elif page == "Research Figures":
        render_page_header("Research Figures", "High-resolution figures ready for publication.")
        render_research_figures()"""

content = re.sub(r'def main\(\):[\s\S]*?render_research_figures\(\)', main_code, content)

with open("app/dashboard.py", "w") as f:
    f.write(content)

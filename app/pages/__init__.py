"""
FedCare Pages Module
====================
Modular page renderers mapped to the 5 MedXChAln structural patterns.
"""

from app.pages.command_center import render_command_center
from app.pages.project_overview import render_project_overview
from app.pages.network_topology import render_network_topology
from app.pages.training_console import render_training_console
from app.pages.live_stream import render_live_stream
from app.pages.communication_cost import render_communication_cost
from app.pages.secure_aggregation import render_secure_aggregation
from app.pages.attack_defense import render_attack_defense
from app.pages.privacy_utility import render_privacy_utility
from app.pages.non_iid_analysis import render_non_iid_analysis
from app.pages.risk_calculator import render_risk_calculator
from app.pages.feature_importance import render_feature_importance
from app.pages.model_comparison import render_model_comparison
from app.pages.data_explorer import render_data_explorer
from app.pages.hospital_deep_dive import render_hospital_deep_dive
from app.pages.experiment_timeline import render_experiment_timeline
from app.pages.research_figures import render_research_figures
from app.pages.anamnesis_record import render_anamnesis_record

__all__ = [
    "render_command_center",
    "render_project_overview",
    "render_network_topology",
    "render_training_console",
    "render_live_stream",
    "render_communication_cost",
    "render_secure_aggregation",
    "render_attack_defense",
    "render_privacy_utility",
    "render_non_iid_analysis",
    "render_risk_calculator",
    "render_feature_importance",
    "render_model_comparison",
    "render_data_explorer",
    "render_hospital_deep_dive",
    "render_experiment_timeline",
    "render_research_figures",
    "render_anamnesis_record",
]

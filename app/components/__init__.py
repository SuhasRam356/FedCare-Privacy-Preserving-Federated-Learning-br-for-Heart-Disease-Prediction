"""
FedCare Components Package
"""

from app.components.theme import (
    inject_clinical_theme,
    HOSPITAL_COLORS,
    STRATEGY_COLORS,
    CHART_COLORS,
    PLOTLY_LAYOUT_DEFAULTS,
)

from app.components.ui_kit import (
    icon_svg,
    svg_world_map,
    svg_traceability_graph,
    svg_integrity_bars,
    svg_compliance_radar,
    status_pill,
    fc_card,
    fc_kpi_card,
    fc_progress_item,
    fc_table,
    fc_button,
    fc_fab,
    render_header,
    render_sidebar_header,
    render_sidebar_footer,
)

from app.components.charts import (
    apply_theme,
    create_convergence_chart,
    create_network_topology,
    create_attack_heatmap,
    create_radar_chart,
    create_gauge,
)

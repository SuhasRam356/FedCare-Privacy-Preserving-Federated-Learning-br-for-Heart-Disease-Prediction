"""
FedCare Enhanced Chart Builders
================================
Premium Plotly chart factory functions with consistent theming.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.components.theme import PLOTLY_LAYOUT_DEFAULTS, HOSPITAL_COLORS, STRATEGY_COLORS, CHART_COLORS


def apply_theme(fig: go.Figure, **overrides) -> go.Figure:
    """Apply the premium FedCare theme to any Plotly figure."""
    layout = {**PLOTLY_LAYOUT_DEFAULTS, **overrides}
    fig.update_layout(**layout)
    return fig


def create_convergence_chart(rounds_df, metric_col, display_name, show_hospitals=True, max_round=None):
    """Create an enhanced convergence chart with optional per-hospital overlays."""
    plot_df = rounds_df
    if max_round is not None:
        plot_df = rounds_df[rounds_df["round"] <= max_round]

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.62, 0.38],
        subplot_titles=(
            f"<span style='color:#f9fafb;font-size:14px'>Global {display_name} Convergence</span>",
            "<span style='color:#f9fafb;font-size:14px'>Per-Hospital AUC</span>",
        ),
        horizontal_spacing=0.08,
    )

    # Global metric with gradient fill
    fig.add_trace(
        go.Scatter(
            x=plot_df["round"], y=plot_df[metric_col],
            mode="lines+markers",
            name=f"Global {display_name}",
            line=dict(width=3, color="#6366f1", shape="spline"),
            marker=dict(size=7, color="#6366f1", line=dict(width=2, color="#818cf8")),
            fill="tozeroy" if metric_col != "test_loss" else None,
            fillcolor="rgba(99, 102, 241, 0.08)",
        ),
        row=1, col=1,
    )

    # Per-hospital AUC overlays
    if show_hospitals and metric_col == "auc":
        for i in range(1, 7):
            hosp_col = f"hosp_{i}_auc"
            if hosp_col in plot_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=plot_df["round"], y=plot_df[hosp_col],
                        mode="lines",
                        name=f"Hospital {i}",
                        line=dict(width=1.5, color=HOSPITAL_COLORS[i - 1], dash="dash"),
                        opacity=0.65,
                    ),
                    row=1, col=1,
                )

    # Hospital AUC bar chart at final round
    last_row = plot_df.iloc[-1]
    hosp_aucs, hosp_labels = [], []
    for i in range(1, 7):
        hosp_col = f"hosp_{i}_auc"
        if hosp_col in last_row.index:
            hosp_aucs.append(float(last_row[hosp_col]))
            hosp_labels.append(f"H{i}")

    if hosp_aucs:
        fig.add_trace(
            go.Bar(
                x=hosp_labels, y=hosp_aucs,
                marker=dict(
                    color=HOSPITAL_COLORS[:len(hosp_aucs)],
                    line=dict(width=1, color="rgba(255,255,255,0.15)"),
                    cornerradius=4,
                ),
                name="Hospital AUC",
                text=[f"{v:.4f}" for v in hosp_aucs],
                textposition="outside",
                textfont=dict(size=10, color="#9ca3af", family="JetBrains Mono"),
            ),
            row=1, col=2,
        )

    fig = apply_theme(fig, height=460)
    fig.update_xaxes(title_text="Communication Round", row=1, col=1)
    fig.update_yaxes(title_text=display_name, row=1, col=1)
    fig.update_xaxes(title_text="Hospital", row=1, col=2)
    if hosp_aucs:
        fig.update_yaxes(
            title_text="AUC", row=1, col=2,
            range=[min(hosp_aucs) - 0.03, max(hosp_aucs) + 0.03],
        )
    return fig


def create_network_topology(hospital_stats):
    """Create an enhanced network topology visualization."""
    fig = go.Figure()

    # Central server
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode="markers+text",
        marker=dict(
            size=60, color="#6366f1", symbol="diamond",
            line=dict(width=3, color="#818cf8"),
            opacity=0.9,
        ),
        text=["FedCare<br>Server"],
        textposition="bottom center",
        textfont=dict(size=11, color="#e0e7ff", family="Inter"),
        name="Central Server",
        hoverinfo="text",
        hovertext=(
            "<b>FedCare Aggregation Server</b><br>"
            "Strategies: FedAvg · FedProx · Krum<br>"
            "Trimmed Mean · Coordinate Median"
        ),
    ))

    n_hospitals = len(hospital_stats)
    angles = np.linspace(0, 2 * np.pi, n_hospitals, endpoint=False) - np.pi / 2
    radius = 3.2

    for idx, (_, row) in enumerate(hospital_stats.iterrows()):
        x = radius * np.cos(angles[idx])
        y = radius * np.sin(angles[idx])

        # Animated connection line
        fig.add_trace(go.Scatter(
            x=[0, x], y=[0, y],
            mode="lines",
            line=dict(width=1.5, color=f"rgba({_hex_to_rgb(HOSPITAL_COLORS[idx])}, 0.25)", dash="dot"),
            showlegend=False, hoverinfo="skip",
        ))

        # Data flow arrow annotation
        mid_x, mid_y = x * 0.55, y * 0.55
        fig.add_annotation(
            x=mid_x, y=mid_y,
            text="⟷", showarrow=False,
            font=dict(size=10, color=f"rgba({_hex_to_rgb(HOSPITAL_COLORS[idx])}, 0.5)"),
        )

        # Hospital node
        prevalence_pct = row["Prevalence"] * 100
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers+text",
            marker=dict(
                size=40 + row["Samples"] / 100,
                color=HOSPITAL_COLORS[idx],
                line=dict(width=2, color="rgba(255,255,255,0.2)"),
                opacity=0.85,
            ),
            text=[f"H{row['ID']}"],
            textposition="middle center",
            textfont=dict(size=13, color="white", family="Inter"),
            name=row["Hospital"],
            hoverinfo="text",
            hovertext=(
                f"<b>{row['Hospital']}</b><br>"
                f"Patients: {row['Samples']:,}<br>"
                f"Prevalence: {prevalence_pct:.1f}%<br>"
                f"Avg Age: {row['Avg_Age']:.1f}<br>"
                f"Avg Cholesterol: {row['Avg_Cholesterol']:.0f} mg/dL"
            ),
        ))

    fig = apply_theme(fig, height=500, showlegend=False)
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, range=[-5, 5])
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, range=[-5, 5])
    fig.update_layout(
        annotations=list(fig.layout.annotations) + [
            dict(
                x=0, y=1.0,
                text="<b>Encrypted Parameters Only</b><br>No raw patient data transmitted",
                showarrow=False,
                font=dict(size=9, color="#6b7280"),
            )
        ]
    )
    return fig


def create_attack_heatmap(pivot_df):
    """Create a premium heatmap for attack x defense matrix."""
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns.tolist(),
        y=pivot_df.index.tolist(),
        colorscale=[
            [0.0, "#7f1d1d"],
            [0.3, "#dc2626"],
            [0.5, "#f59e0b"],
            [0.7, "#22c55e"],
            [1.0, "#059669"],
        ],
        text=np.round(pivot_df.values, 4),
        texttemplate="%{text}",
        textfont=dict(size=13, color="white", family="JetBrains Mono"),
        hoverongaps=False,
        colorbar=dict(
            title=dict(text="AUC", font=dict(color="#9ca3af")),
            tickfont=dict(color="#9ca3af"),
        ),
    ))
    fig = apply_theme(fig, height=380)
    fig.update_layout(
        xaxis_title="Defense Strategy",
        yaxis_title="Attack Scenario",
    )
    return fig


def create_radar_chart(categories, values_dict, title=""):
    """Create a radar/spider chart for strategy comparison."""
    fig = go.Figure()

    for name, values in values_dict.items():
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=name,
            fillcolor=f"rgba({_hex_to_rgb(STRATEGY_COLORS[list(values_dict.keys()).index(name) % len(STRATEGY_COLORS)])}, 0.1)",
            line=dict(
                color=STRATEGY_COLORS[list(values_dict.keys()).index(name) % len(STRATEGY_COLORS)],
                width=2,
            ),
        ))

    fig = apply_theme(fig, height=420)
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                gridcolor="rgba(75, 85, 99, 0.2)",
                linecolor="rgba(75, 85, 99, 0.2)",
                tickfont=dict(color="#6b7280", size=9),
            ),
            angularaxis=dict(
                gridcolor="rgba(75, 85, 99, 0.2)",
                linecolor="rgba(75, 85, 99, 0.15)",
                tickfont=dict(color="#9ca3af", size=11),
            ),
        ),
        title=dict(text=title, font=dict(size=14, color="#f9fafb")),
    )
    return fig


def create_gauge(value, title="", color="#6366f1"):
    """Create a premium gauge chart for risk calculator."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number=dict(
            suffix="%",
            font=dict(size=52, color="#f9fafb", family="JetBrains Mono"),
        ),
        title=dict(text=title, font=dict(size=14, color="#9ca3af")),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor="#4b5563", dtick=25),
            bar=dict(color=color, thickness=0.25),
            bgcolor="rgba(17, 24, 39, 0.5)",
            borderwidth=1,
            bordercolor="rgba(75, 85, 99, 0.3)",
            steps=[
                dict(range=[0, 25], color="rgba(16, 185, 129, 0.12)"),
                dict(range=[25, 50], color="rgba(245, 158, 11, 0.12)"),
                dict(range=[50, 75], color="rgba(239, 68, 68, 0.12)"),
                dict(range=[75, 100], color="rgba(220, 38, 38, 0.15)"),
            ],
            threshold=dict(
                line=dict(color=color, width=4),
                thickness=0.8,
                value=value * 100,
            ),
        ),
    ))
    fig = apply_theme(fig, height=320)
    fig.update_layout(margin=dict(l=30, r=30, t=60, b=20))
    return fig


def _hex_to_rgb(hex_color: str) -> str:
    """Convert hex color to comma-separated RGB string."""
    hex_color = hex_color.lstrip("#")
    return f"{int(hex_color[0:2], 16)}, {int(hex_color[2:4], 16)}, {int(hex_color[4:6], 16)}"

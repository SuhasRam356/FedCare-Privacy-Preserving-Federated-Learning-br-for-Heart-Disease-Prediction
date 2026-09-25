"""
FedCare Clinical Research Theme System
======================================
Refined, trustworthy muted-dark theme appropriate for a medical-AI thesis defense.
Focuses on readability, accessibility, and professional presentation.
"""

import streamlit as st

def inject_clinical_theme():
    """Inject the clean, clinical muted-dark CSS theme."""
    st.markdown(
        """
        <style>
        /* ═══════════════════════════════════════════════════════════════
           GOOGLE FONTS
           ═══════════════════════════════════════════════════════════════ */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Roboto+Mono:wght@400;500&display=swap');

        /* ═══════════════════════════════════════════════════════════════
           CSS CUSTOM PROPERTIES (Design Tokens)
           ═══════════════════════════════════════════════════════════════ */
        :root {
            /* Primary palette - Clinical Blue / Teal */
            --fc-primary: #3b82f6; /* Blue 500 */
            --fc-primary-light: #60a5fa; /* Blue 400 */
            --fc-primary-dark: #2563eb; /* Blue 600 */
            
            /* Accent - Medical Teal */
            --fc-accent: #0d9488; /* Teal 600 */
            --fc-accent-light: #14b8a6; /* Teal 500 */
            
            /* Semantic */
            --fc-success: #10b981; /* Emerald 500 */
            --fc-warning: #f59e0b; /* Amber 500 */
            --fc-danger: #ef4444; /* Red 500 */
            
            /* Surfaces (Muted Dark) */
            --fc-bg-deep: #0f172a; /* Slate 900 */
            --fc-bg: #1e293b; /* Slate 800 */
            --fc-bg-elevated: #334155; /* Slate 700 */
            --fc-surface: #1e293b;
            
            /* Text */
            --fc-text: #f8fafc; /* Slate 50 */
            --fc-text-secondary: #cbd5e1; /* Slate 300 */
            --fc-text-muted: #94a3b8; /* Slate 400 */
            
            /* Borders */
            --fc-border: #334155; /* Slate 700 */
            --fc-border-hover: #475569; /* Slate 600 */
            
            /* Radii */
            --fc-radius-sm: 4px;
            --fc-radius: 8px;
            --fc-radius-lg: 12px;
            
            /* Shadows */
            --fc-shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.2);
            --fc-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        }

        /* ═══════════════════════════════════════════════════════════════
           BASE STYLES
           ═══════════════════════════════════════════════════════════════ */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
            color: var(--fc-text);
        }

        .stApp {
            background-color: var(--fc-bg-deep);
        }

        /* Hide Streamlit chrome */
        #MainMenu, header, footer { visibility: hidden; }
        [data-testid="stToolbar"] { display: none !important; }
        .stDeployButton { display: none !important; }
        
        /* Clean Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: var(--fc-bg) !important;
            border-right: 1px solid var(--fc-border) !important;
        }
        [data-testid="stSidebarCollapseButton"] { display: none !important; }

        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            color: var(--fc-text);
            font-weight: 600 !important;
            letter-spacing: -0.025em;
        }

        p, span, div {
            color: var(--fc-text-secondary);
            line-height: 1.6;
        }
        
        strong {
            color: var(--fc-text);
            font-weight: 600;
        }

        /* Metrics */
        div[data-testid="stMetricValue"] {
            font-family: 'Roboto Mono', monospace !important;
            font-size: 1.8rem !important;
            font-weight: 500 !important;
            color: var(--fc-primary-light) !important;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.85rem !important;
            color: var(--fc-text-muted) !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* Buttons */
        .stButton>button {
            background-color: var(--fc-bg-elevated);
            color: var(--fc-text);
            border: 1px solid var(--fc-border);
            border-radius: var(--fc-radius);
            font-weight: 500;
            transition: all 0.2s;
        }
        .stButton>button:hover {
            border-color: var(--fc-primary-light);
            color: var(--fc-primary-light);
        }
        
        /* Primary Button */
        .stButton>button[kind="primary"] {
            background-color: var(--fc-primary);
            border-color: var(--fc-primary);
            color: #ffffff;
        }
        .stButton>button[kind="primary"]:hover {
            background-color: var(--fc-primary-light);
            border-color: var(--fc-primary-light);
        }

        /* Expander */
        .streamlit-expanderHeader {
            background-color: var(--fc-bg-elevated) !important;
            border-radius: var(--fc-radius) !important;
            border: 1px solid var(--fc-border) !important;
            color: var(--fc-text) !important;
        }
        .streamlit-expanderContent {
            border: 1px solid var(--fc-border) !important;
            border-top: none !important;
            border-radius: 0 0 var(--fc-radius) var(--fc-radius) !important;
            background-color: var(--fc-bg) !important;
        }

        /* Dataframes / Tables */
        .stDataFrame, .stTable {
            background-color: var(--fc-bg);
            border-radius: var(--fc-radius);
            border: 1px solid var(--fc-border);
        }
        
        /* Custom Cards */
        .fc-card {
            background-color: var(--fc-bg);
            border: 1px solid var(--fc-border);
            border-radius: var(--fc-radius);
            padding: 1.5rem;
            margin-bottom: 1rem;
        }
        
        /* Headers */
        .fc-page-title {
            font-size: 2.25rem !important;
            font-weight: 700 !important;
            margin-bottom: 0.25rem;
            color: var(--fc-text);
        }
        .fc-page-subtitle {
            font-size: 1.1rem;
            color: var(--fc-text-muted);
            margin-bottom: 2rem;
            margin-top: 0;
        }
        
        /* Divider */
        hr {
            border-top: 1px solid var(--fc-border);
            margin: 2rem 0;
        }
        
        /* Alerts */
        .stAlert {
            border-radius: var(--fc-radius);
        }
        
        /* Tags */
        .fc-tag {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-right: 0.5rem;
            margin-bottom: 0.5rem;
            background-color: var(--fc-bg-elevated);
            border: 1px solid var(--fc-border);
        }
        
        /* Risk bands */
        .risk-low { color: var(--fc-success); }
        .risk-moderate { color: var(--fc-warning); }
        .risk-high { color: var(--fc-danger); }
        </style>
        """,
        unsafe_allow_html=True
    )

def render_page_header(title: str, subtitle: str):
    """Render a consistent page header."""
    st.markdown(f"""
        <div style="padding-bottom: 1rem;">
            <h1 class="fc-page-title">{title}</h1>
            <p class="fc-page-subtitle">{subtitle}</p>
        </div>
    """, unsafe_allow_html=True)

# ── Python Styling Constants ──────────────────────────────────────────

# 6 distinct, color-blind-safe colors for hospitals (clinical theme)
HOSPITAL_COLORS = [
    "#3b82f6", # Blue
    "#10b981", # Emerald
    "#f59e0b", # Amber
    "#ef4444", # Red
    "#8b5cf6", # Violet
    "#06b6d4", # Cyan
]

STRATEGY_COLORS = {
    "FedAvg": "#3b82f6",
    "FedProx": "#10b981",
    "Krum": "#f59e0b",
    "Trimmed Mean": "#8b5cf6",
    "Coordinate Median": "#06b6d4",
    "Centralized": "#ef4444"
}

CHART_COLORS = {
    "primary": "#3b82f6",
    "secondary": "#cbd5e1",
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444"
}

PLOTLY_LAYOUT_DEFAULTS = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#f8fafc", family="'Inter', sans-serif", size=12),
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.15,
        xanchor="center",
        x=0.5,
        bgcolor="rgba(30, 41, 59, 0.8)", 
        bordercolor="rgba(71, 85, 105, 0.5)",
        borderwidth=1, 
        font=dict(size=11, color="#cbd5e1")
    ),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor="rgba(71, 85, 105, 0.3)", zerolinecolor="rgba(71, 85, 105, 0.5)"),
    yaxis=dict(gridcolor="rgba(71, 85, 105, 0.3)", zerolinecolor="rgba(71, 85, 105, 0.5)"),
)


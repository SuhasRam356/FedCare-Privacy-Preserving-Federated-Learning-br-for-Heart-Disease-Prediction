"""
FedCare Clinical SaaS Theme — MedXChAln Design System
======================================================
Light, professional, clinical-grade theme for a healthcare-AI dashboard.
Provides the global CSS injection, design tokens, and style constants.
"""

from __future__ import annotations
import streamlit as st


# ══════════════════════════════════════════════════════════════════════
#                    GLOBAL CSS INJECTION
# ══════════════════════════════════════════════════════════════════════

def render_page_header(title: str, subtitle: str):
    """Render a unified MedXChAln style header."""
    st.markdown(f"""
    <div class="fc-header-bar">
        <div class="fc-header-left">
            <h1 class="fc-header-title">{title}</h1>
            <p class="fc-header-subtitle">{subtitle}</p>
        </div>
        <div class="fc-header-right">
            <div class="fc-header-search">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#94A3B8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
                <input type="text" class="fc-search-input" placeholder="Search...">
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def inject_clinical_theme(active_page_key: str = ""):
    """Inject the full light clinical MedXChAln CSS theme."""
    active_nav_css = ""
    if active_page_key:
        active_nav_css = f"""
        /* CONDITIONAL ACTIVE NAV BUTTON */
        div.st-key-{active_page_key} button {{
            background-color: #EFF6FF !important;
            color: #0F5BB6 !important;
            font-weight: 700 !important;
            border-radius: 8px !important;
        }}
        div.st-key-{active_page_key} button svg,
        div.st-key-{active_page_key} button span,
        div.st-key-{active_page_key} button p {{
            color: #0F5BB6 !important;
            stroke: #0F5BB6 !important;
            font-weight: 700 !important;
        }}
        """

    css_body = f"""
        /* ═══════════════════════════════════════════════════════════════
        GOOGLE FONTS — Inter
        ═══════════════════════════════════════════════════════════════ */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');


        /* ═══════════════════════════════════════════════════════════════
           CSS CUSTOM PROPERTIES — MedXChAln Light Clinical Tokens
           ═══════════════════════════════════════════════════════════════ */
        :root {{
            --fc-bg: #F6FAFF;
            --fc-surface: #FFFFFF;
            --fc-sidebar-bg: #FFFFFF;
            --fc-border: #E5EAF2;
            --fc-border-subtle: #F1F5F9;
            --fc-primary: #0F5BB6;
            --fc-primary-dark: #0B4A93;
            --fc-primary-light: #EFF6FF;
            --fc-navy-panel: #0B2545;
            --fc-navy-gradient: linear-gradient(135deg, #0B2545 0%, #133B70 100%);
            --fc-text: #0F172A;
            --fc-text-secondary: #64748B;
            --fc-text-muted: #94A3B8;
            --fc-success: #15803D;
            --fc-success-bg: #DCFCE7;
            --fc-warning: #B45309;
            --fc-warning-bg: #FEF3C7;
            --fc-danger: #B91C1C;
            --fc-danger-bg: #FEE2E2;
            --fc-radius: 14px;
            --fc-radius-sm: 8px;
            --fc-radius-pill: 999px;
            --fc-shadow: 0 2px 8px rgba(15,23,42,0.04), 0 1px 2px rgba(15,23,42,0.02);
            --fc-shadow-md: 0 4px 12px rgba(15,23,42,0.06), 0 2px 4px rgba(15,23,42,0.04);
            --fc-shadow-lg: 0 10px 25px -5px rgba(15,91,182,0.1), 0 8px 10px -6px rgba(15,91,182,0.06);
        }}

        /* ═══════════════════════════════════════════════════════════════
           BASE STYLES
           ═══════════════════════════════════════════════════════════════ */
        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
            color: var(--fc-text);
            -webkit-font-smoothing: antialiased;
        }}

        .stApp {{
            background-color: var(--fc-bg) !important;
        }}

        /* Hide Streamlit chrome */
        #MainMenu, header, footer {{ visibility: hidden; }}
        [data-testid="stToolbar"] {{ display: none !important; }}
        .stDeployButton {{ display: none !important; }}
        div[data-testid="stDecoration"] {{ display: none !important; }}

        /* Main content layout spacing */
        .block-container {{
            padding-top: 1.5rem !important;
            padding-bottom: 3rem !important;
            max-width: 1400px !important;
        }}

        /* ═══════════════════════════════════════════════════════════════
           TYPOGRAPHY
           ═══════════════════════════════════════════════════════════════ */
        h1, h2, h3, h4, h5, h6 {{
            color: var(--fc-text) !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em;
            -webkit-text-fill-color: var(--fc-text) !important;
            background: none !important;
        }}

        p, li, td, th {{
            color: var(--fc-text-secondary);
            line-height: 1.5;
        }}

        /* ═══════════════════════════════════════════════════════════════
           SIDEBAR — MedXChAln Exact Specification
           ═══════════════════════════════════════════════════════════════ */
        [data-testid="stSidebar"] {{
            background-color: var(--fc-sidebar-bg) !important;
            border-right: 1px solid var(--fc-border) !important;
            min-width: 250px !important;
            max-width: 270px !important;
            box-shadow: none !important;
        }}
        [data-testid="stSidebarCollapseButton"] {{ display: none !important; }}

        /* Top Logo Row */
        .fc-sidebar-logo-row {{
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 8px 4px 16px 4px;
            border-bottom: 1px solid var(--fc-border);
            margin-bottom: 14px;
        }}
        .fc-logo-square {{
            width: 38px;
            height: 38px;
            border-radius: 10px;
            background: var(--fc-primary);
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 2px 6px rgba(15,91,182,0.25);
            flex-shrink: 0;
        }}
        .fc-wordmark {{
            font-size: 1.2rem;
            font-weight: 800;
            color: var(--fc-text);
            letter-spacing: -0.02em;
            display: flex;
            align-items: center;
            gap: 6px;
        }}

        .fc-sidebar-logo-row {{
            transition: opacity 120ms ease;
        }}
        .fc-sidebar-logo-row:hover {{
            opacity: 0.85;
        }}

        /* Identity Block */
        .fc-identity-block {{
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 10px 12px;
            background: #F8FAFC;
            border: 1px solid var(--fc-border);
            border-radius: 12px;
            margin-bottom: 16px;
            transition: all 120ms ease;
        }}
        .fc-identity-block:hover {{
            background: #EFF6FF;
            border-color: #BFDBFE;
        }}
        .fc-avatar-circle {{
            width: 34px;
            height: 34px;
            border-radius: 50%;
            background: #EFF6FF;
            border: 1.5px solid #BFDBFE;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }}
        .fc-identity-text {{
            line-height: 1.25;
        }}
        .fc-identity-name {{
            font-size: 0.84rem;
            font-weight: 700;
            color: var(--fc-text);
        }}
        .fc-identity-role {{
            font-size: 0.72rem;
            color: var(--fc-text-secondary);
        }}

        /* Sidebar Status Card & Links */
        .fc-sidebar-status-card {{
            background: #F8FAFC;
            border: 1px solid var(--fc-border);
            border-radius: 10px;
            padding: 10px 12px;
            margin-top: 20px;
            margin-bottom: 12px;
            transition: all 120ms ease;
        }}
        .fc-sidebar-status-card:hover {{
            background: #EFF6FF;
            border-color: #BFDBFE;
        }}
        .fc-sidebar-links {{
            display: flex;
            flex-direction: column;
            gap: 4px;
            margin-top: 8px;
        }}
        a.fc-sidebar-link-row {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.8rem;
            font-weight: 500;
            color: #64748B !important;
            text-decoration: none !important;
            padding: 6px 8px;
            border-radius: 6px;
            transition: all 120ms ease;
            cursor: pointer;
        }}
        a.fc-sidebar-link-row:hover {{
            color: var(--fc-primary) !important;
            background: #EFF6FF !important;
        }}

        /* Sidebar Section Header */
        .fc-sidebar-section-title {{
            font-size: 0.68rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--fc-text-muted);
            margin: 14px 4px 6px 4px;
        }}

        /* Sidebar Nav Buttons */
        [data-testid="stSidebar"] div[class*="st-key-nav_"] button {{
            background: transparent !important;
            border: none !important;
            color: var(--fc-text-secondary) !important;
            font-weight: 500 !important;
            font-size: 0.88rem !important;
            text-align: left !important;
            justify-content: flex-start !important;
            padding: 10px 12px !important;
            border-radius: 8px !important;
            height: 44px !important;
            transition: all 120ms ease !important;
            box-shadow: none !important;
            margin-bottom: 3px !important;
            width: 100% !important;
        }}
        [data-testid="stSidebar"] div[class*="st-key-nav_"] button:hover {{
            background: #F1F5F9 !important;
            color: var(--fc-text) !important;
        }}

        {active_nav_css}

        /* Sidebar Status Card */
        .fc-sidebar-status-card {{
            background: var(--fc-primary-light);
            border: 1px solid #DBEAFE;
            border-radius: 10px;
            padding: 12px 14px;
            margin-top: 20px;
            margin-bottom: 12px;
        }}
        .fc-status-pulse-dot {{
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #10B981;
            display: inline-block;
            box-shadow: 0 0 0 2px rgba(16,185,129,0.25);
            animation: fc-pulse 2s ease-in-out infinite;
        }}
        @keyframes fc-pulse {{
            0%, 100% {{ opacity: 1; transform: scale(1); }}
            50% {{ opacity: 0.5; transform: scale(0.9); }}
        }}

        /* Sidebar Links (Settings / Support) */
        .fc-sidebar-links {{
            border-top: 1px solid var(--fc-border);
            padding-top: 10px;
            margin-top: 10px;
        }}
        .fc-sidebar-link-row {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 6px;
            color: var(--fc-text-secondary);
            font-size: 0.84rem;
            font-weight: 500;
            border-radius: 6px;
            cursor: pointer;
            transition: all 120ms ease;
        }}
        .fc-sidebar-link-row:hover {{
            background: #F1F5F9;
            color: var(--fc-text);
        }}

        /* ═══════════════════════════════════════════════════════════════
           HEADER BAR COMPONENT
           ═══════════════════════════════════════════════════════════════ */
        .fc-header-bar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 4px 0 16px 0;
            border-bottom: 1px solid var(--fc-border);
            margin-bottom: 22px;
            width: 100%;
            flex-wrap: wrap;
            gap: 16px;
        }}
        .fc-header-left {{
            display: flex;
            flex-direction: column;
            gap: 2px;
            flex-shrink: 0;
            min-width: 260px;
        }}
        .fc-header-title {{
            font-size: 1.55rem !important;
            font-weight: 800 !important;
            color: var(--fc-text) !important;
            margin: 0 !important;
            letter-spacing: -0.02em;
            white-space: nowrap !important;
        }}

        .fc-header-subtitle {{
            font-size: 0.85rem !important;
            color: var(--fc-text-secondary) !important;
            margin: 0 !important;
        }}
        .fc-header-right {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .fc-header-search {{
            display: flex;
            align-items: center;
            background: #F1F5F9;
            border-radius: 999px;
            padding: 7px 16px;
            width: 220px;
            border: 1px solid transparent;
            transition: all 150ms ease;
        }}
        .fc-header-search:focus-within {{
            border-color: #CBD5E1;
            background: #FFFFFF;
        }}
        .fc-search-input {{
            border: none;
            background: transparent;
            outline: none;
            font-size: 0.82rem;
            color: var(--fc-text);
            width: 100%;
        }}
        .fc-search-input::placeholder {{
            color: #94A3B8;
        }}
        .fc-header-meta {{
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            line-height: 1.2;
            padding: 0 4px;
        }}
        .fc-header-meta-label {{
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--fc-text-secondary);
        }}
        .fc-header-meta-sync {{
            font-size: 0.68rem;
            color: var(--fc-text-muted);
        }}
        .fc-header-icons {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .fc-icon-btn {{
            width: 34px;
            height: 34px;
            border-radius: 8px;
            border: 1px solid var(--fc-border);
            background: #FFFFFF;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 120ms ease;
            text-decoration: none !important;
            color: #64748B !important;
        }}
        .fc-icon-btn:hover {{
            background: #F8FAFC;
            border-color: #CBD5E1;
            color: var(--fc-primary) !important;
        }}
        .fc-header-action-btn {{
            background: var(--fc-primary);
            color: #FFFFFF !important;
            text-decoration: none !important;
            border: none;
            border-radius: 8px;
            padding: 8px 16px;
            font-size: 0.85rem;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            cursor: pointer;
            box-shadow: 0 2px 6px rgba(15,91,182,0.25);
            transition: all 120ms ease;
        }}
        .fc-header-action-btn:hover {{
            background: var(--fc-primary-dark);
            color: #FFFFFF !important;
            text-decoration: none !important;
        }}
        a.fc-fab {{
            text-decoration: none !important;
            color: #FFFFFF !important;
        }}

        /* ═══════════════════════════════════════════════════════════════
           CARDS & SURFACES
           ═══════════════════════════════════════════════════════════════ */
        .fc-card {{
            background: var(--fc-surface);
            border: 1px solid var(--fc-border);
            border-radius: var(--fc-radius);
            padding: 22px 24px;
            box-shadow: var(--fc-shadow);
            margin-bottom: 16px;
            transition: box-shadow 0.2s ease;
        }}
        .fc-card-navy {{
            background: var(--fc-navy-gradient);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: var(--fc-radius);
            padding: 24px 26px;
            box-shadow: var(--fc-shadow-md);
            margin-bottom: 16px;
            color: #FFFFFF;
            position: relative;
            overflow: hidden;
        }}
        .fc-card-navy h3, .fc-card-navy h4 {{
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
        }}
        .fc-card-navy p, .fc-card-navy span {{
            color: rgba(255,255,255,0.8) !important;
        }}

        .fc-stat-card {{
            background: var(--fc-surface);
            border: 1px solid var(--fc-border);
            border-radius: var(--fc-radius);
            padding: 18px 20px;
            box-shadow: var(--fc-shadow);
            margin-bottom: 16px;
            transition: transform 150ms ease, box-shadow 150ms ease, border-color 150ms ease;
        }}
        .fc-stat-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 18px rgba(15,91,182,0.12);
            border-color: #CBD5E1;
        }}
        a.fc-kpi-link {{
            text-decoration: none !important;
            color: inherit !important;
            display: block;
        }}
        a.fc-kpi-link:hover {{
            text-decoration: none !important;
            color: inherit !important;
        }}

        /* ═══════════════════════════════════════════════════════════════
           STATUS PILLS
           ═══════════════════════════════════════════════════════════════ */
        .fc-pill {{
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 3px 10px;
            border-radius: var(--fc-radius-pill);
            font-size: 0.68rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            line-height: 1.5;
            white-space: nowrap;
        }}
        .fc-pill-dot {{
            width: 6px;
            height: 6px;
            border-radius: 50%;
            display: inline-block;
        }}
        .fc-pill-success {{ background: var(--fc-success-bg); color: var(--fc-success); }}
        .fc-pill-success .fc-pill-dot {{ background: var(--fc-success); }}

        .fc-pill-warning {{ background: var(--fc-warning-bg); color: var(--fc-warning); }}
        .fc-pill-warning .fc-pill-dot {{ background: var(--fc-warning); }}

        .fc-pill-danger  {{ background: var(--fc-danger-bg);  color: var(--fc-danger); }}
        .fc-pill-danger .fc-pill-dot  {{ background: var(--fc-danger); }}

        .fc-pill-info    {{ background: var(--fc-primary-light); color: var(--fc-primary); }}
        .fc-pill-info .fc-pill-dot    {{ background: var(--fc-primary); }}

        .fc-pill-blockchain {{ background: #EEF2FF; color: #4338CA; border: 1px solid #E0E7FF; }}
        .fc-pill-blockchain .fc-pill-dot {{ background: #4338CA; }}

        .fc-pill-navy    {{ background: var(--fc-navy-panel); color: #FFFFFF; }}

        /* ═══════════════════════════════════════════════════════════════
           BUTTONS & FAB
           ═══════════════════════════════════════════════════════════════ */
        .fc-btn {{
            border-radius: 8px;
            padding: 9px 18px;
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 120ms ease;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }}
        .fc-btn-primary {{
            background: var(--fc-primary);
            color: #FFFFFF !important;
            border: none;
            box-shadow: 0 2px 4px rgba(15,91,182,0.2);
        }}
        .fc-btn-primary:hover {{
            background: var(--fc-primary-dark);
        }}
        .fc-btn-secondary {{
            background: #FFFFFF;
            color: var(--fc-text) !important;
            border: 1px solid #CBD5E1;
        }}
        .fc-btn-secondary:hover {{
            background: #F8FAFC;
            border-color: #94A3B8;
        }}
        .fc-btn-danger-outline {{
            background: #FEF2F2;
            color: var(--fc-danger) !important;
            border: 1px solid #FCA5A5;
        }}
        .fc-btn-navy-cta {{
            background: #FFFFFF;
            color: var(--fc-navy-panel) !important;
            border: none;
            font-weight: 700;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
        }}

        .fc-fab {{
            position: fixed;
            bottom: 30px;
            right: 32px;
            width: 52px;
            height: 52px;
            border-radius: 50%;
            background: var(--fc-primary);
            color: #FFFFFF;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 6px 16px rgba(15,91,182,0.35);
            cursor: pointer;
            z-index: 999;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        .fc-fab:hover {{
            transform: scale(1.06);
            box-shadow: 0 8px 20px rgba(15,91,182,0.45);
        }}

        /* ═══════════════════════════════════════════════════════════════
           DROPDOWN / SELECTBOX FIX (Prevents invisible text)
           ═══════════════════════════════════════════════════════════════ */
        div[data-baseweb="select"] {{
            background-color: #FFFFFF !important;
            color: #0F172A !important;
            border-radius: 8px !important;
        }}
        div[data-baseweb="select"] * {{
            color: #0F172A !important;
            background-color: transparent !important;
        }}
        div[data-baseweb="popover"],
        ul[role="listbox"],
        li[role="option"] {{
            background-color: #FFFFFF !important;
            color: #0F172A !important;
        }}
        li[role="option"]:hover {{
            background-color: #EFF6FF !important;
            color: #0F5BB6 !important;
        }}

        /* Form Inputs */
        div[data-testid="stTextInput"] input,
        div[data-testid="stNumberInput"] input,
        div[data-testid="stTextArea"] textarea {{
            background-color: #F8FAFC !important;
            border: 1px solid var(--fc-border) !important;
            border-radius: 8px !important;
            color: var(--fc-text) !important;
            font-size: 0.88rem !important;
        }}
        div[data-testid="stTextInput"] input:focus,
        div[data-testid="stNumberInput"] input:focus,
        div[data-testid="stTextArea"] textarea:focus {{
            border-color: var(--fc-primary) !important;
            background-color: #FFFFFF !important;
            box-shadow: 0 0 0 2px rgba(15,91,182,0.12) !important;
        }}

        /* Slider */
        div[data-testid="stSlider"] {{
            color: var(--fc-primary) !important;
        }}

        /* ═══════════════════════════════════════════════════════════════
           STREAMLIT NATIVE BUTTON RESTYLE
           ═══════════════════════════════════════════════════════════════ */
        .stButton > button {{
            background-color: var(--fc-primary) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 9px 20px !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
            transition: all 120ms ease !important;
            box-shadow: 0 2px 4px rgba(15,91,182,0.2) !important;
        }}
        .stButton > button:hover {{
            background-color: var(--fc-primary-dark) !important;
        }}

        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 6px;
            background: #F1F5F9;
            border-radius: 10px;
            padding: 4px;
            border: 1px solid var(--fc-border);
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px;
            color: var(--fc-text-secondary);
            font-weight: 600;
            padding: 8px 18px;
            font-size: 0.84rem;
        }}
        .stTabs [aria-selected="true"] {{
            background: #FFFFFF !important;
            color: var(--fc-primary) !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }}

        /* Scrollbar */
        ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
        ::-webkit-scrollbar-track {{ background: transparent; }}
        ::-webkit-scrollbar-thumb {{ background: #CBD5E1; border-radius: 3px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: #94A3B8; }}
        """
    clean_css = "\n".join(line.strip() for line in css_body.splitlines() if line.strip())
    st.markdown(f"<style>\n{clean_css}\n</style>", unsafe_allow_html=True)



# ══════════════════════════════════════════════════════════════════════
#                    STYLE CONSTANTS & PLOTLY DEFAULTS
# ══════════════════════════════════════════════════════════════════════

HOSPITAL_COLORS = [
    "#0F5BB6",  # Primary blue
    "#16A34A",  # Green
    "#D97706",  # Amber
    "#DC2626",  # Red
    "#7C3AED",  # Violet
    "#0891B2",  # Cyan
]

STRATEGY_COLORS = {
    "FedAvg": "#0F5BB6",
    "FedProx": "#16A34A",
    "Krum": "#D97706",
    "Trimmed Mean": "#7C3AED",
    "Coordinate Median": "#0891B2",
    "Centralized": "#DC2626"
}

CHART_COLORS = {
    "primary": "#0F5BB6",
    "secondary": "#94A3B8",
    "success": "#16A34A",
    "warning": "#D97706",
    "danger": "#DC2626"
}

PLOTLY_LAYOUT_DEFAULTS = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#0F172A", family="'Inter', sans-serif", size=12),
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.15,
        xanchor="center",
        x=0.5,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#E5EAF2",
        borderwidth=1,
        font=dict(size=11, color="#64748B")
    ),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(
        gridcolor="rgba(226,232,240,0.6)",
        zerolinecolor="#E5EAF2",
        tickfont=dict(color="#64748B", size=11),
        title_font=dict(color="#64748B", size=12),
    ),
    yaxis=dict(
        gridcolor="rgba(226,232,240,0.6)",
        zerolinecolor="#E5EAF2",
        tickfont=dict(color="#64748B", size=11),
        title_font=dict(color="#64748B", size=12),
    ),
)

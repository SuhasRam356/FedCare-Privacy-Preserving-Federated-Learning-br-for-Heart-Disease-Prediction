"""
FedCare Premium Theme System
=============================
Enterprise-grade dark theme with glassmorphism, animated gradients,
micro-interactions, and curated color palette.
"""

import streamlit as st


def inject_premium_css():
    """Inject the complete premium CSS theme with animations and glassmorphism."""
    st.markdown(
        """
        <style>
        /* ═══════════════════════════════════════════════════════════════
           GOOGLE FONTS
           ═══════════════════════════════════════════════════════════════ */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

        /* ═══════════════════════════════════════════════════════════════
           CSS CUSTOM PROPERTIES (Design Tokens)
           ═══════════════════════════════════════════════════════════════ */
        :root {
            /* Primary palette - Indigo / Violet */
            --fc-primary: #6366f1;
            --fc-primary-light: #818cf8;
            --fc-primary-dark: #4f46e5;
            --fc-primary-glow: rgba(99, 102, 241, 0.15);
            --fc-primary-glow-strong: rgba(99, 102, 241, 0.3);

            /* Accent - Cyan */
            --fc-accent: #06b6d4;
            --fc-accent-light: #22d3ee;
            --fc-accent-glow: rgba(6, 182, 212, 0.15);

            /* Semantic */
            --fc-success: #10b981;
            --fc-success-glow: rgba(16, 185, 129, 0.15);
            --fc-warning: #f59e0b;
            --fc-warning-glow: rgba(245, 158, 11, 0.15);
            --fc-danger: #ef4444;
            --fc-danger-glow: rgba(239, 68, 68, 0.15);

            /* Surfaces */
            --fc-bg-deep: #030712;
            --fc-bg: #0a0f1a;
            --fc-bg-elevated: #111827;
            --fc-surface: rgba(17, 24, 39, 0.7);
            --fc-surface-hover: rgba(31, 41, 55, 0.8);
            --fc-glass: rgba(17, 24, 39, 0.5);
            --fc-glass-border: rgba(99, 102, 241, 0.12);

            /* Text */
            --fc-text: #f9fafb;
            --fc-text-secondary: #9ca3af;
            --fc-text-muted: #6b7280;
            --fc-text-dim: #4b5563;

            /* Borders */
            --fc-border: rgba(75, 85, 99, 0.3);
            --fc-border-hover: rgba(99, 102, 241, 0.4);

            /* Shadows */
            --fc-shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.3);
            --fc-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
            --fc-shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.5);
            --fc-shadow-glow: 0 0 20px var(--fc-primary-glow);

            /* Transitions */
            --fc-ease: cubic-bezier(0.4, 0, 0.2, 1);
            --fc-duration: 200ms;
            --fc-duration-slow: 350ms;

            /* Radii */
            --fc-radius-sm: 6px;
            --fc-radius: 10px;
            --fc-radius-lg: 14px;
            --fc-radius-xl: 20px;

            /* Hospital brand colors */
            --fc-h1: #f43f5e;
            --fc-h2: #f59e0b;
            --fc-h3: #10b981;
            --fc-h4: #3b82f6;
            --fc-h5: #8b5cf6;
            --fc-h6: #ec4899;
        }

        /* ═══════════════════════════════════════════════════════════════
           KEYFRAME ANIMATIONS
           ═══════════════════════════════════════════════════════════════ */

        @keyframes shimmer {
            0% { background-position: -200% 0; }
            100% { background-position: 200% 0; }
        }

        @keyframes pulse-glow {
            0%, 100% { box-shadow: 0 0 8px var(--fc-primary-glow); }
            50% { box-shadow: 0 0 20px var(--fc-primary-glow-strong); }
        }

        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-4px); }
        }

        @keyframes gradient-shift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        @keyframes fade-in-up {
            from { opacity: 0; transform: translateY(12px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes spin-slow {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }

        @keyframes border-dance {
            0% { border-color: var(--fc-primary); }
            33% { border-color: var(--fc-accent); }
            66% { border-color: var(--fc-success); }
            100% { border-color: var(--fc-primary); }
        }

        /* ═══════════════════════════════════════════════════════════════
           BASE STYLES
           ═══════════════════════════════════════════════════════════════ */

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }

        .stApp {
            background: var(--fc-bg-deep);
            background-image:
                radial-gradient(ellipse at 20% 0%, rgba(99, 102, 241, 0.06) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 100%, rgba(6, 182, 212, 0.04) 0%, transparent 50%);
        }

        /* Hide Streamlit chrome */
        #MainMenu, header, footer { visibility: hidden; }
        .stDeployButton { display: none; }

        /* ═══════════════════════════════════════════════════════════════
           SIDEBAR
           ═══════════════════════════════════════════════════════════════ */

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #060a14 0%, #0c1222 100%) !important;
            border-right: 1px solid var(--fc-border);
            backdrop-filter: blur(20px);
        }

        section[data-testid="stSidebar"] .stMarkdown {
            color: var(--fc-text-secondary);
        }

        /* ═══════════════════════════════════════════════════════════════
           METRIC CARDS (Glassmorphic)
           ═══════════════════════════════════════════════════════════════ */

        div[data-testid="stMetric"] {
            background: var(--fc-glass);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--fc-glass-border);
            border-radius: var(--fc-radius-lg);
            padding: 20px 24px;
            box-shadow: var(--fc-shadow);
            transition: all var(--fc-duration) var(--fc-ease);
            animation: fade-in-up 0.4s var(--fc-ease) backwards;
            position: relative;
            overflow: hidden;
        }

        div[data-testid="stMetric"]::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, var(--fc-primary), var(--fc-accent), var(--fc-primary));
            background-size: 200% 100%;
            animation: gradient-shift 3s ease infinite;
            opacity: 0;
            transition: opacity var(--fc-duration) var(--fc-ease);
        }

        div[data-testid="stMetric"]:hover {
            border-color: var(--fc-border-hover);
            transform: translateY(-2px);
            box-shadow: var(--fc-shadow-lg), var(--fc-shadow-glow);
        }

        div[data-testid="stMetric"]:hover::before {
            opacity: 1;
        }

        div[data-testid="stMetric"] label {
            color: var(--fc-text-secondary) !important;
            font-size: 0.75rem !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.8px;
        }

        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: var(--fc-text) !important;
            font-size: 1.9rem !important;
            font-weight: 800 !important;
            font-family: 'JetBrains Mono', monospace !important;
            letter-spacing: -0.5px;
        }

        div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
            font-size: 0.8rem !important;
            font-weight: 500 !important;
        }

        /* Stagger animation for metric cards */
        div[data-testid="stMetric"]:nth-child(1) { animation-delay: 0.05s; }
        div[data-testid="stMetric"]:nth-child(2) { animation-delay: 0.1s; }
        div[data-testid="stMetric"]:nth-child(3) { animation-delay: 0.15s; }
        div[data-testid="stMetric"]:nth-child(4) { animation-delay: 0.2s; }
        div[data-testid="stMetric"]:nth-child(5) { animation-delay: 0.25s; }

        /* ═══════════════════════════════════════════════════════════════
           HEADERS
           ═══════════════════════════════════════════════════════════════ */

        h1, h2, h3, h4, h5, h6 {
            color: var(--fc-text) !important;
            font-weight: 700 !important;
        }

        h1 {
            font-size: 2.4rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.8px;
            background: linear-gradient(135deg, var(--fc-text) 0%, var(--fc-primary-light) 50%, var(--fc-accent-light) 100%);
            background-size: 200% 200%;
            animation: gradient-shift 4s ease infinite;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        h2 {
            font-size: 1.6rem !important;
            letter-spacing: -0.3px;
            color: var(--fc-text) !important;
        }

        /* ═══════════════════════════════════════════════════════════════
           TAB STYLING
           ═══════════════════════════════════════════════════════════════ */

        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background: var(--fc-glass);
            backdrop-filter: blur(12px);
            border-radius: var(--fc-radius);
            padding: 4px;
            border: 1px solid var(--fc-border);
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: var(--fc-radius-sm);
            color: var(--fc-text-secondary);
            font-weight: 500;
            padding: 8px 16px;
            transition: all var(--fc-duration) var(--fc-ease);
        }

        .stTabs [data-baseweb="tab"]:hover {
            color: var(--fc-text);
            background: rgba(99, 102, 241, 0.08);
        }

        .stTabs [aria-selected="true"] {
            background: var(--fc-primary-glow) !important;
            color: var(--fc-primary-light) !important;
            font-weight: 600;
            border-bottom: 2px solid var(--fc-primary) !important;
        }

        /* ═══════════════════════════════════════════════════════════════
           BUTTONS
           ═══════════════════════════════════════════════════════════════ */

        .stButton > button {
            background: linear-gradient(135deg, var(--fc-primary) 0%, var(--fc-primary-dark) 100%) !important;
            color: white !important;
            border: 1px solid rgba(129, 140, 248, 0.3) !important;
            border-radius: var(--fc-radius) !important;
            padding: 10px 24px !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            letter-spacing: 0.2px;
            transition: all var(--fc-duration) var(--fc-ease) !important;
            box-shadow: 0 2px 8px rgba(99, 102, 241, 0.25) !important;
        }

        .stButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 16px rgba(99, 102, 241, 0.4) !important;
        }

        .stButton > button:active {
            transform: translateY(0) !important;
        }

        /* ═══════════════════════════════════════════════════════════════
           FORM INPUTS
           ═══════════════════════════════════════════════════════════════ */

        .stSelectbox > div > div,
        .stNumberInput > div > div > input,
        .stTextInput > div > div > input {
            border-color: var(--fc-border) !important;
            background-color: var(--fc-bg-elevated) !important;
            color: var(--fc-text) !important;
            border-radius: var(--fc-radius-sm) !important;
            transition: border-color var(--fc-duration) var(--fc-ease) !important;
        }

        .stSelectbox > div > div:focus-within,
        .stNumberInput > div > div > input:focus,
        .stTextInput > div > div > input:focus {
            border-color: var(--fc-primary) !important;
            box-shadow: 0 0 0 2px var(--fc-primary-glow) !important;
        }

        /* Slider */
        .stSlider > div {
            color: var(--fc-text) !important;
        }

        /* ═══════════════════════════════════════════════════════════════
           EXPANDER
           ═══════════════════════════════════════════════════════════════ */

        .streamlit-expanderHeader {
            background: var(--fc-glass) !important;
            border: 1px solid var(--fc-border) !important;
            border-radius: var(--fc-radius) !important;
            color: var(--fc-text) !important;
            font-weight: 500;
            backdrop-filter: blur(8px);
            transition: all var(--fc-duration) var(--fc-ease);
        }

        .streamlit-expanderHeader:hover {
            border-color: var(--fc-border-hover) !important;
            background: var(--fc-surface-hover) !important;
        }

        /* ═══════════════════════════════════════════════════════════════
           DATAFRAME
           ═══════════════════════════════════════════════════════════════ */

        .stDataFrame {
            border-radius: var(--fc-radius) !important;
            overflow: hidden;
        }

        [data-testid="stDataFrame"] > div {
            border-radius: var(--fc-radius) !important;
            border: 1px solid var(--fc-border) !important;
        }

        /* ═══════════════════════════════════════════════════════════════
           DIVIDERS
           ═══════════════════════════════════════════════════════════════ */

        hr {
            border-color: var(--fc-border) !important;
            opacity: 0.5;
        }

        /* ═══════════════════════════════════════════════════════════════
           SCROLLBAR
           ═══════════════════════════════════════════════════════════════ */

        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: transparent;
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(99, 102, 241, 0.25);
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(99, 102, 241, 0.4);
        }

        /* ═══════════════════════════════════════════════════════════════
           CUSTOM COMPONENT CLASSES
           ═══════════════════════════════════════════════════════════════ */

        /* Hero badge */
        .fc-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: var(--fc-primary-glow);
            color: var(--fc-primary-light);
            border: 1px solid rgba(99, 102, 241, 0.2);
            padding: 5px 14px;
            border-radius: 20px;
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            animation: fade-in-up 0.3s var(--fc-ease);
        }

        .fc-badge::before {
            content: '';
            width: 6px;
            height: 6px;
            background: var(--fc-primary);
            border-radius: 50%;
            animation: pulse-glow 2s ease-in-out infinite;
        }

        /* Glass card */
        .fc-glass-card {
            background: var(--fc-glass);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--fc-glass-border);
            border-radius: var(--fc-radius-lg);
            padding: 24px;
            transition: all var(--fc-duration) var(--fc-ease);
            animation: fade-in-up 0.4s var(--fc-ease) backwards;
        }

        .fc-glass-card:hover {
            border-color: var(--fc-border-hover);
            box-shadow: var(--fc-shadow-glow);
        }

        /* Stat highlight */
        .fc-stat {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            font-size: 2rem;
            background: linear-gradient(135deg, var(--fc-primary-light), var(--fc-accent-light));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        /* Phase pill */
        .fc-phase-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            border-radius: 24px;
            font-size: 0.85rem;
            font-weight: 600;
            transition: all var(--fc-duration) var(--fc-ease);
        }

        .fc-phase-pill:hover {
            transform: scale(1.02);
        }

        .fc-phase-1 { background: rgba(99, 102, 241, 0.1); color: #818cf8; border: 1px solid rgba(99, 102, 241, 0.2); }
        .fc-phase-2 { background: rgba(6, 182, 212, 0.1); color: #22d3ee; border: 1px solid rgba(6, 182, 212, 0.2); }
        .fc-phase-3 { background: rgba(16, 185, 129, 0.1); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.2); }
        .fc-phase-4 { background: rgba(239, 68, 68, 0.1); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.2); }
        .fc-phase-5 { background: rgba(168, 85, 247, 0.1); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.2); }

        /* Hospital node cards */
        .fc-hospital-card {
            text-align: center;
            padding: 20px;
            background: var(--fc-glass);
            backdrop-filter: blur(12px);
            border: 1px solid var(--fc-glass-border);
            border-radius: var(--fc-radius-lg);
            transition: all var(--fc-duration-slow) var(--fc-ease);
            animation: fade-in-up 0.5s var(--fc-ease) backwards;
            position: relative;
            overflow: hidden;
        }

        .fc-hospital-card::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: var(--card-accent, var(--fc-primary));
            opacity: 0;
            transition: opacity var(--fc-duration) var(--fc-ease);
        }

        .fc-hospital-card:hover {
            border-color: var(--fc-border-hover);
            transform: translateY(-3px);
            box-shadow: var(--fc-shadow-lg);
        }

        .fc-hospital-card:hover::after {
            opacity: 1;
        }

        /* Risk gauge colors */
        .fc-risk-low { color: var(--fc-success); font-weight: 800; }
        .fc-risk-moderate { color: var(--fc-warning); font-weight: 800; }
        .fc-risk-high { color: var(--fc-danger); font-weight: 800; }
        .fc-risk-very-high { color: #dc2626; font-weight: 800; text-shadow: 0 0 20px rgba(220, 38, 38, 0.3); }

        /* Section separator with gradient */
        .fc-separator {
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--fc-border), var(--fc-primary-glow), var(--fc-border), transparent);
            margin: 2rem 0;
        }

        /* Animated gradient text */
        .fc-gradient-text {
            background: linear-gradient(135deg, #818cf8, #22d3ee, #34d399, #818cf8);
            background-size: 300% 300%;
            animation: gradient-shift 4s ease infinite;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        /* Info card with icon */
        .fc-info-card {
            display: flex;
            align-items: flex-start;
            gap: 16px;
            padding: 20px;
            background: var(--fc-glass);
            border: 1px solid var(--fc-glass-border);
            border-radius: var(--fc-radius);
            backdrop-filter: blur(12px);
        }

        .fc-info-card .fc-info-icon {
            width: 40px;
            height: 40px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
            flex-shrink: 0;
        }

        /* Notification / toast style */
        div[data-testid="stAlert"] {
            background: var(--fc-glass) !important;
            border: 1px solid var(--fc-glass-border) !important;
            border-radius: var(--fc-radius) !important;
            backdrop-filter: blur(12px);
        }

        /* Toggle styling */
        .stToggle label span {
            color: var(--fc-text-secondary) !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Plotly Theme Defaults ─────────────────────────────────────────────

PLOTLY_LAYOUT_DEFAULTS = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(
        color="#f9fafb",
        family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
        size=12,
    ),
    legend=dict(
        bgcolor="rgba(17, 24, 39, 0.7)",
        bordercolor="rgba(75, 85, 99, 0.3)",
        borderwidth=1,
        font=dict(size=11, color="#9ca3af"),
    ),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(
        gridcolor="rgba(75, 85, 99, 0.15)",
        zerolinecolor="rgba(75, 85, 99, 0.2)",
    ),
    yaxis=dict(
        gridcolor="rgba(75, 85, 99, 0.15)",
        zerolinecolor="rgba(75, 85, 99, 0.2)",
    ),
)

# Color sequences
HOSPITAL_COLORS = ["#f43f5e", "#f59e0b", "#10b981", "#3b82f6", "#8b5cf6", "#ec4899"]
STRATEGY_COLORS = ["#6366f1", "#06b6d4", "#10b981", "#f59e0b", "#f43f5e", "#8b5cf6"]
CHART_COLORS = ["#6366f1", "#22d3ee", "#10b981", "#f59e0b", "#ef4444", "#a855f7", "#ec4899"]

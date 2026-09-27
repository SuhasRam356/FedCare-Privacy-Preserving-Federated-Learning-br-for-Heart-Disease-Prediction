"""
FedCare UI Kit — MedXChAln Design System Component Library
==========================================================
Clinical SaaS design system component library matching MedXChAln specifications:
- Pure HTML/CSS blocks rendered via st.markdown(unsafe_allow_html=True)
- Feather/Lucide SVG line icon set (no emojis)
- Illustrative SVG graphics (World map with live nodes, Traceability pipeline, Integrity bars)
- Status pills, KPI cards, Navy hero cards, Data tables, Buttons, Headers, and Sidebar
- 5 Structural Grid Layout Templates
"""

from __future__ import annotations
import html
from typing import Any, Dict, List, Optional, Tuple


def clean_html(raw_html: str) -> str:
    """
    Remove all indentation and empty lines from HTML strings before passing to Streamlit markdown.
    In Markdown/CommonMark:
    - Any line indented with 4+ spaces after a blank line is parsed as an INDENTED CODE BLOCK (<pre><code>).
    - An empty line breaks an HTML block, causing subsequent lines to be parsed as markdown/code blocks.
    By stripping whitespace from every line and joining non-empty lines, EVERY line starts at column 0.
    This guarantees 100% pure HTML interpretation by Streamlit with zero code block leaks.
    """
    if not raw_html:
        return ""
    lines = [line.strip() for line in str(raw_html).strip().splitlines() if line.strip()]
    return "\n".join(lines)


def html_block(raw_html: str) -> None:
    """Safely render HTML in Streamlit without risk of Markdown 4-space code block bug."""
    import streamlit as st
    st.markdown(clean_html(raw_html), unsafe_allow_html=True)



# ══════════════════════════════════════════════════════════════════════
#                     1. INLINE SVG ICON LIBRARY
# ══════════════════════════════════════════════════════════════════════

def icon_svg(name: str, size: int = 18, color: str = "currentColor", stroke_width: float = 2.0) -> str:
    """Return an inline SVG string for a crisp Feather/Lucide line icon."""
    s = str(size)
    sw = str(stroke_width)
    base = f'<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; display: inline-block;">'
    
    paths = {
        "overview": '<rect width="7" height="9" x="3" y="3" rx="1"/><rect width="7" height="5" x="14" y="3" rx="1"/><rect width="7" height="9" x="14" y="12" rx="1"/><rect width="7" height="5" x="3" y="16" rx="1"/>',
        "dashboard": '<rect width="7" height="9" x="3" y="3" rx="1"/><rect width="7" height="5" x="14" y="3" rx="1"/><rect width="7" height="9" x="14" y="12" rx="1"/><rect width="7" height="5" x="3" y="16" rx="1"/>',
        "user": '<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
        "anamnesis": '<path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><path d="M15 2H9a1 1 0 0 0-1 1v2a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1V3a1 1 0 0 0-1-1Z"/><path d="M12 11h4"/><path d="M12 16h4"/><path d="M8 11h.01"/><path d="M8 16h.01"/>',
        "diagnosis": '<path d="m14 12-8.5 8.5a2.12 2.12 0 1 1-3-3L11 9"/><path d="M12 14l4-4"/><path d="m18 10 4-4"/><path d="m14 6 4 4"/><path d="m21 3-3 3"/>',
        "stethoscope": '<path d="M4.8 2.3A.3.3 0 1 0 5 2H4a2 2 0 0 0-2 2v5a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6V4a2 2 0 0 0-2-2h-1a.2.2 0 1 0 .3.3"/><path d="M8 15v1a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6v-4"/><circle cx="20" cy="10" r="2"/>',
        "microscope": '<path d="M6 18h8"/><path d="M3 22h18"/><path d="m14 22-.72-3.6A6 6 0 0 0 7.4 14H6"/><circle cx="12" cy="6" r="3"/><path d="M9 13.5A7.5 7.5 0 0 0 16.5 21"/>',
        "model": '<path d="M12 2v4"/><path d="m4.93 4.93 2.83 2.83"/><path d="M2 12h4"/><path d="m4.93 19.07 2.83-2.83"/><path d="M12 22v-4"/><path d="m19.07 19.07-2.83-2.83"/><path d="M22 12h-4"/><path d="m19.07 4.93-2.83 2.83"/><circle cx="12" cy="12" r="3"/>',
        "cpu": '<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/>',
        "hospital": '<path d="M12 6V2H4v20h16V6h-8Z"/><path d="M8 6h.01"/><path d="M16 10h.01"/><path d="M12 10h.01"/><path d="M8 10h.01"/><path d="M12 14h.01"/><path d="M8 14h.01"/><path d="M12 18h8"/><path d="M8 18h.01"/>',
        "network": '<rect width="6" height="6" x="9" y="2" rx="1"/><rect width="6" height="6" x="2" y="16" rx="1"/><rect width="6" height="6" x="16" y="16" rx="1"/><path d="M5 16v-3a1 1 0 0 1 1-1h12a1 1 0 0 1 1 1v3"/><path d="M12 12V8"/>',
        "shield": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/>',
        "shield_check": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/>',
        "search": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
        "refresh": '<path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/><path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16"/><path d="M16 16h5v5"/>',
        "bell": '<path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9"/><path d="M10.3 21a1.94 1.94 0 0 0 3.4 0"/>',
        "bell_dot": '<path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9"/><path d="M10.3 21a1.94 1.94 0 0 0 3.4 0"/><circle cx="18" cy="4" r="3" fill="#0F5BB6" stroke="#FFFFFF" stroke-width="1.5"/>',
        "settings": '<path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/>',
        "help": '<circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><path d="M12 17h.01"/>',
        "plus": '<path d="M5 12h14"/><path d="M12 5v14"/>',
        "check": '<path d="M20 6 9 17l-5-5"/>',
        "check_circle": '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
        "alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
        "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/><path d="M3 12c0 1.66 4 3 9 3s9-1.34 9-3"/>',
        "lock": '<rect width="18" height="11" x="3" y="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>',
        "activity": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
        "download": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/>',
        "zap": '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
        "terminal": '<polyline points="4 17 10 11 4 5"/><line x1="12" x2="20" y1="19" y2="19"/>',
        "clipboard": '<rect width="8" height="4" x="8" y="2" rx="1" ry="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/>',
        "filter": '<polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/>',
        "chevron_right": '<path d="m9 18 6-6-6-6"/>',
        "chevron_left": '<path d="m15 18-6-6 6-6"/>',
        "printer": '<polyline points="6 9 6 2 18 2 18 9"/><path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"/><rect width="12" height="8" x="6" y="14"/>',
        "share": '<circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" x2="15.42" y1="13.51" y2="17.49"/><line x1="15.41" x2="8.59" y1="6.51" y2="10.49"/>',
        "layers": '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
        "trending_up": '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',
        "sliders": '<line x1="4" x2="4" y1="21" y2="14"/><line x1="4" x2="4" y1="10" y2="3"/><line x1="12" x2="12" y1="21" y2="12"/><line x1="12" x2="12" y1="8" y2="3"/><line x1="20" x2="20" y1="21" y2="16"/><line x1="20" x2="20" y1="12" y2="3"/><line x1="1" x2="7" y1="14" y2="14"/><line x1="9" x2="15" y1="8" y2="8"/><line x1="17" x2="23" y1="16" y2="16"/>',
        "bar_chart": '<line x1="12" x2="12" y1="20" y2="10"/><line x1="18" x2="18" y1="20" y2="4"/><line x1="6" x2="6" y1="20" y2="16"/>',
        "star": '<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>',
        "globe": '<circle cx="12" cy="12" r="10"/><line x1="2" x2="22" y1="12" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>',
        "heart": '<path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/>',
        "node_tree": '<circle cx="12" cy="5" r="3"/><circle cx="5" cy="19" r="3"/><circle cx="19" cy="19" r="3"/><line x1="12" y1="8" x2="5" y2="16"/><line x1="12" y1="8" x2="19" y2="16"/>',
        "chain": '<path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>'
    }
    
    path_data = paths.get(name.lower(), paths["activity"])
    return f"{base}{path_data}</svg>"


# ══════════════════════════════════════════════════════════════════════
#                     2. ILLUSTRATIVE SVG GRAPHICS
# ══════════════════════════════════════════════════════════════════════

def svg_world_map(badge_text: str = "LIVE NODES") -> str:
    """Return stylized World Map SVG with pulsing live node markers matching Screenshot 4."""
    raw = f"""
    <div style="position:relative; width:100%; border-radius:12px; overflow:hidden; background:#5B6471; padding:18px 12px; display:flex; flex-direction:column; align-items:center; justify-content:center;">
        <svg viewBox="0 0 480 230" width="100%" height="auto" style="max-height:180px;" fill="none" xmlns="http://www.w3.org/2000/svg">
            <!-- Simplified landmass paths -->
            <path d="M50 55 C65 40, 110 35, 125 50 C140 65, 145 100, 115 110 C90 120, 60 115, 50 85 Z" fill="#7A8596" opacity="0.85"/>
            <path d="M100 125 C115 130, 130 160, 120 185 C110 200, 95 190, 85 160 C80 140, 90 130, 100 125 Z" fill="#7A8596" opacity="0.85"/>
            <path d="M210 40 C240 30, 275 35, 290 55 C285 75, 250 85, 230 75 C215 70, 205 55, 210 40 Z" fill="#7A8596" opacity="0.85"/>
            <path d="M225 90 C250 85, 270 100, 265 140 C260 170, 240 180, 230 155 C220 130, 215 105, 225 90 Z" fill="#7A8596" opacity="0.85"/>
            <path d="M295 45 C350 35, 420 50, 430 85 C420 120, 360 115, 330 100 C305 85, 290 60, 295 45 Z" fill="#7A8596" opacity="0.85"/>
            <path d="M380 140 C410 135, 435 155, 425 180 C405 195, 375 180, 370 160 C365 145, 375 140, 380 140 Z" fill="#7A8596" opacity="0.85"/>
            
            <!-- Radiating Node Pulses -->
            <!-- North America / Minnesota -->
            <circle cx="95" cy="75" r="9" fill="#0F5BB6" opacity="0.35"/>
            <circle cx="95" cy="75" r="4.5" fill="#38BDF8"/>
            <circle cx="95" cy="75" r="2" fill="#FFFFFF"/>
            
            <!-- Europe / London / Berlin -->
            <circle cx="235" cy="55" r="9" fill="#0F5BB6" opacity="0.35"/>
            <circle cx="235" cy="55" r="4.5" fill="#38BDF8"/>
            <circle cx="235" cy="55" r="2" fill="#FFFFFF"/>
            
            <!-- Asia / Singapore -->
            <circle cx="365" cy="115" r="9" fill="#0F5BB6" opacity="0.35"/>
            <circle cx="365" cy="115" r="4.5" fill="#38BDF8"/>
            <circle cx="365" cy="115" r="2" fill="#FFFFFF"/>
            
            <!-- Australia -->
            <circle cx="400" cy="160" r="7" fill="#0F5BB6" opacity="0.35"/>
            <circle cx="400" cy="160" r="3.5" fill="#38BDF8"/>
            
            <!-- Connection Lines -->
            <path d="M95 75 Q165 40 235 55" stroke="#38BDF8" stroke-width="1.2" stroke-dasharray="3 3" opacity="0.75"/>
            <path d="M235 55 Q300 70 365 115" stroke="#38BDF8" stroke-width="1.2" stroke-dasharray="3 3" opacity="0.75"/>
        </svg>
        <div style="position:absolute; bottom:14px; background:#0F5BB6; color:#FFFFFF; font-size:10px; font-weight:700; letter-spacing:0.06em; padding:4px 12px; border-radius:999px; box-shadow:0 2px 6px rgba(0,0,0,0.25);">
            ● {badge_text}
        </div>
    </div>
    """
    return clean_html(raw)


def svg_traceability_graph() -> str:
    """Return connected pipeline graphic with magnifying inspection glass matching Screenshot 5."""
    raw = """
    <div style="width:100%; display:flex; align-items:center; justify-content:center; padding:10px;">
        <svg viewBox="0 0 340 180" width="100%" height="auto" style="max-height:160px;" fill="none" xmlns="http://www.w3.org/2000/svg">
            <!-- Background grid points -->
            <line x1="30" y1="90" x2="310" y2="90" stroke="#E2E8F0" stroke-width="1.5" stroke-dasharray="4 4"/>
            
            <!-- Multi-colored pipeline path -->
            <path d="M 40 85 L 80 120 L 125 105 L 160 80 L 195 125 L 245 110 L 290 140" stroke="#64748B" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>
            
            <!-- Nodes along the path -->
            <circle cx="40" cy="85" r="10" fill="#F59E0B" stroke="#FFFFFF" stroke-width="3"/>
            <circle cx="80" cy="120" r="10" fill="#EF4444" stroke="#FFFFFF" stroke-width="3"/>
            <circle cx="125" cy="105" r="10" fill="#0EA5E9" stroke="#FFFFFF" stroke-width="3"/>
            <circle cx="160" cy="80" r="10" fill="#10B981" stroke="#FFFFFF" stroke-width="3"/>
            <circle cx="195" cy="125" r="10" fill="#3B82F6" stroke="#FFFFFF" stroke-width="3"/>
            <circle cx="245" cy="110" r="10" fill="#94A3B8" stroke="#FFFFFF" stroke-width="3"/>
            <circle cx="290" cy="140" r="10" fill="#64748B" stroke="#FFFFFF" stroke-width="3"/>
            
            <!-- Magnifying Glass hovering on verified node (195, 125) -->
            <g transform="translate(20, -10)">
                <circle cx="175" cy="135" r="28" fill="#FFFFFF" fill-opacity="0.88" stroke="#CBD5E1" stroke-width="6"/>
                <circle cx="175" cy="135" r="20" fill="#EFF6FF" stroke="#3B82F6" stroke-width="3"/>
                <circle cx="175" cy="135" r="8" fill="#1D4ED8"/>
                <line x1="155" y1="155" x2="128" y2="182" stroke="#64748B" stroke-width="9" stroke-linecap="round"/>
            </g>
        </svg>
    </div>
    """
    return clean_html(raw)


def svg_integrity_bars(heights: List[int] = None, color: str = "#93C5FD") -> str:
    """Return inline mini bar chart cluster for Trust / Integrity score card matching Screenshot 5."""
    if heights is None:
        heights = [35, 55, 95, 45, 80, 60]
    bars_html = ""
    w = 14
    gap = 8
    max_h = 75
    total_w = len(heights) * (w + gap)
    for i, h in enumerate(heights):
        x = i * (w + gap)
        y = max_h - h
        bars_html += f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="{color}" opacity="0.85"/>'
    
    raw = f"""
    <svg viewBox="0 0 {total_w} {max_h}" width="100%" height="{max_h}px" fill="none" xmlns="http://www.w3.org/2000/svg">
        {bars_html}
    </svg>
    """
    return clean_html(raw)


def svg_compliance_radar() -> str:
    """Return subtle cyber compliance radar / circuit graphic for the navy CTA card matching Screenshot 2."""
    raw = """
    <div style="position:absolute; right:15px; top:10px; width:130px; height:130px; pointer-events:none; opacity:0.35;">
        <svg viewBox="0 0 100 100" width="100%" height="100%" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="50" cy="50" r="42" stroke="#60A5FA" stroke-width="1.5" stroke-dasharray="3 3"/>
            <circle cx="50" cy="50" r="30" stroke="#38BDF8" stroke-width="2"/>
            <circle cx="50" cy="50" r="18" stroke="#93C5FD" stroke-width="1.5"/>
            <circle cx="50" cy="50" r="5" fill="#60A5FA"/>
            <line x1="50" y1="8" x2="50" y2="92" stroke="#60A5FA" stroke-width="1" opacity="0.5"/>
            <line x1="8" y1="50" x2="92" y2="50" stroke="#60A5FA" stroke-width="1" opacity="0.5"/>
        </svg>
    </div>
    """
    return clean_html(raw)



# ══════════════════════════════════════════════════════════════════════
#                     3. STATUS PILL COMPONENT
# ══════════════════════════════════════════════════════════════════════

def status_pill(text: str, kind: str = "success", dot: bool = True, icon_name: Optional[str] = None) -> str:
    """
    Produce a rounded-pill status label color-coded by semantic meaning.
    Kinds:
        'success'    -> Light green, dark green text (Active, Verified, Ready)
        'warning'    -> Light amber, amber text (Review, Outlier)
        'danger'     -> Light red, red text (Flagged, Inactive, Attack)
        'info'       -> Light blue, blue text (Pending, Latest, Sync)
        'blockchain' -> Indigo, blockchain verified
        'navy'       -> Dark navy, white text
    """
    kind = kind.lower().strip()
    icon_html = ""
    if icon_name:
        icon_html = icon_svg(icon_name, size=12, stroke_width=2.5) + " "
    elif dot and kind != "navy":
        icon_html = '<span class="fc-pill-dot"></span>'
        
    raw = f'<span class="fc-pill fc-pill-{kind}">{icon_html}{html.escape(text)}</span>'
    return clean_html(raw)


# ══════════════════════════════════════════════════════════════════════
#                     4. CARD PRIMITIVES
# ══════════════════════════════════════════════════════════════════════

def fc_card(
    content: str,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    badge: Optional[str] = None,
    right_action: Optional[str] = None,
    navy: bool = False,
    icon_name: Optional[str] = None,
    class_name: str = ""
) -> str:
    """Render a clinical SaaS surface card (white with subtle shadow, or dark navy hero variant)."""
    cls = "fc-card-navy" if navy else "fc-card"
    if class_name:
        cls += f" {class_name}"
        
    header_html = ""
    if title or badge or right_action:
        icon_part = f'<span style="margin-right:8px;">{icon_svg(icon_name, size=18, color="#FFFFFF" if navy else "#0F5BB6")}</span>' if icon_name else ""
        badge_part = f'<span style="margin-left:8px;">{badge}</span>' if badge else ""
        right_part = f'<div>{right_action}</div>' if right_action else ""
        subtitle_part = f'<p class="fc-card-sub" style="margin:2px 0 0 0; font-size:0.8rem; color:{"rgba(255,255,255,0.7)" if navy else "#64748B"};">{subtitle}</p>' if subtitle else ""
        
        header_html = f"""
        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:14px; border-bottom:1px solid {"rgba(255,255,255,0.1)" if navy else "#F1F5F9"}; padding-bottom:10px;">
            <div>
                <div style="display:flex; align-items:center;">
                    {icon_part}
                    <h3 style="margin:0; font-size:1.05rem; font-weight:700; color:{"#FFFFFF" if navy else "#0F172A"}; letter-spacing:-0.01em;">{html.escape(title or "")}</h3>
                    {badge_part}
                </div>
                {subtitle_part}
            </div>
            {right_part}
        </div>
        """
        
    raw = f"""
    <div class="{cls}">
        {header_html}
        <div class="fc-card-body">
            {content}
        </div>
    </div>
    """
    return clean_html(raw)


def default_href_for_kpi(title: str) -> str:
    """Infer the most relevant page route based on KPI metric title."""
    t = (title or "").upper()
    if any(k in t for k in ["HOSPITAL", "NODE", "FACILIT"]):
        return "/?page=Hospital+Deep+Dive"
    elif any(k in t for k in ["MODEL", "UPDATE", "ROUND", "PARAM", "ARCHITECTURE"]):
        return "/?page=Model+Comparison"
    elif any(k in t for k in ["AUC", "ACCURACY", "ROC", "METRIC", "ENSEMBLE", "EQUITY"]):
        return "/?page=Model+Comparison"
    elif any(k in t for k in ["STATUS", "SECURITY", "BLOCKCHAIN", "INTEGRITY", "TAMPER", "LEDGER", "PROOF"]):
        return "/?page=Attack+vs.+Defense"
    elif any(k in t for k in ["PATIENT", "COHORT", "SAMPLE", "CONTRIBUTION", "BIOMARKER"]):
        return "/?page=Data+Explorer"
    elif any(k in t for k in ["PREVALENCE", "DIAGNOSIS", "RISK", "INFERENCE", "SYMPTOM"]):
        return "/?page=Data+Explorer"
    elif any(k in t for k in ["PRIVACY", "EPSILON", "DP", "NOISE", "GUARANTEE"]):
        return "/?page=Privacy-Utility"
    return "/?page=Project+Overview"


def fc_kpi_card(
    title: str,
    value: str,
    subtext: Optional[str] = None,
    badge: Optional[str] = None,
    icon_name: str = "activity",
    icon_bg: str = "#EFF6FF",
    icon_color: str = "#0F5BB6",
    watermark: Optional[str] = None,
    href: Optional[str] = None
) -> str:
    """Render an interactive KPI Metric card matching the 4 top cards from Screenshot 4."""
    icon_markup = icon_svg(icon_name, size=18, color=icon_color, stroke_width=2.2)
    badge_html = f'<div>{badge}</div>' if badge else ""
    sub_html = f'<p style="margin:4px 0 0 0; font-size:0.75rem; color:#64748B; font-weight:500;">{subtext}</p>' if subtext else ""
    watermark_html = f'<div style="position:absolute; right:12px; bottom:8px; opacity:0.12; pointer-events:none;">{icon_svg(watermark, size=46, color=icon_color)}</div>' if watermark else ""
    
    target_href = href or default_href_for_kpi(title)
    
    raw = f"""
    <a href="{target_href}" target="_top" class="fc-kpi-link" title="Navigate to {html.escape(title)} details">
        <div class="fc-stat-card" style="position:relative; overflow:hidden; cursor:pointer;">
            {watermark_html}
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <div style="width:34px; height:34px; border-radius:8px; background:{icon_bg}; display:flex; align-items:center; justify-content:center;">
                    {icon_markup}
                </div>
                {badge_html}
            </div>
            <p style="margin:0 0 4px 0; font-size:0.75rem; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; color:#64748B;">
                {html.escape(title)}
            </p>
            <div style="font-size:1.85rem; font-weight:800; color:#0F172A; line-height:1.15; font-family:'Inter', sans-serif;">
                {html.escape(value)}
            </div>
            {sub_html}
        </div>
    </a>
    """
    return clean_html(raw)


def fc_progress_item(
    label: str,
    value_str: str,
    percentage: float,
    color: str = "#0F5BB6",
    sublabel: Optional[str] = None,
    href: Optional[str] = None
) -> str:
    """Render an entity row with an inline progress bar (for participation/importance lists)."""
    p = max(0.0, min(100.0, float(percentage)))
    sub_part = f'<div style="font-size:0.7rem; color:rgba(255,255,255,0.6);">{sublabel}</div>' if sublabel else ""
    
    inner_content = f"""
    <div style="margin-bottom:14px; transition:opacity 0.15s ease;" {"onmouseover=\"this.style.opacity='0.85'\" onmouseout=\"this.style.opacity='1.0'\"" if href else ""}>
        <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:4px;">
            <div>
                <span style="font-size:0.85rem; font-weight:600;">{html.escape(label)}</span>
                {sub_part}
            </div>
            <span style="font-size:0.85rem; font-weight:700;">{html.escape(value_str)}</span>
        </div>
        <div style="width:100%; height:6px; background:rgba(255,255,255,0.15); border-radius:999px; overflow:hidden;">
            <div style="width:{p}%; height:100%; background:{color}; border-radius:999px; transition: width 0.3s ease;"></div>
        </div>
    </div>
    """
    if href:
        raw = f'<a href="{href}" target="_top" style="text-decoration:none; color:inherit; display:block;" title="Inspect {html.escape(label)}">{inner_content}</a>'
    else:
        raw = inner_content
    return clean_html(raw)



# ══════════════════════════════════════════════════════════════════════
#                     5. CLINICAL DATA TABLE
# ══════════════════════════════════════════════════════════════════════

def fc_table(
    headers: List[str],
    rows: List[List[Any]],
    status_col_idx: Optional[int] = None,
    progress_cols: Optional[Dict[int, str]] = None,
    pager: Optional[Dict[str, Any]] = None,
    actions: Optional[Dict[int, str]] = None,
    chips: Optional[Dict[int, str]] = None
) -> str:
    """
    Render a clean clinical SaaS table matching Screenshots 2, 3, and 5:
    - Uppercase muted headers with subtle border
    - Generous vertical padding and bottom hairlines
    - Monogram avatar chips in leading column
    - Inline thin progress bars for numerical metrics
    - Color-coded status pills
    - Right-aligned actions (e.g. outline buttons)
    - Pager footer
    """
    if progress_cols is None:
        progress_cols = {}
        
    th_cells = "".join(f'<th style="text-align:left; padding:12px 14px; font-size:0.7rem; font-weight:700; color:#64748B; text-transform:uppercase; letter-spacing:0.06em; border-bottom:1px solid #E5EAF2;">{html.escape(h)}</th>' for h in headers)
    
    rows_html = ""
    for r_idx, row in enumerate(rows):
        td_cells = ""
        for c_idx, cell in enumerate(row):
            cell_val = str(cell)
            cell_content = cell_val
            
            # Monogram chip in col 0 if provided or row starts with identifier
            if c_idx == 0 and chips and r_idx in chips:
                chip_txt = chips[r_idx]
                cell_content = f"""
                <div style="display:flex; align-items:center; gap:10px;">
                    <div style="width:32px; height:32px; border-radius:8px; background:#EFF6FF; color:#0F5BB6; display:flex; align-items:center; justify-content:center; font-weight:700; font-size:0.75rem;">
                        {chip_txt}
                    </div>
                    <div>{cell_val}</div>
                </div>
                """
            # Status column
            elif status_col_idx is not None and c_idx == status_col_idx:
                raw_lower = cell_val.lower()
                kind = "success" if any(w in raw_lower for w in ["active", "valid", "commit", "passed", "low"]) else \
                       "danger" if any(w in raw_lower for w in ["inactive", "flag", "alert", "fail", "high"]) else \
                       "warning" if any(w in raw_lower for w in ["pending", "outlier", "moderate", "review"]) else "info"
                cell_content = status_pill(cell_val, kind=kind)
                
            # Progress bar column
            elif c_idx in progress_cols:
                try:
                    num_val = float(cell_val.replace("%", "").replace(",", "").strip())
                    p = min(100.0, max(0.0, num_val))
                    bar_color = progress_cols[c_idx]
                    cell_content = f"""
                    <div style="display:flex; align-items:center; gap:8px;">
                        <span style="font-weight:600; min-width:44px;">{cell_val}</span>
                        <div style="flex-grow:1; max-width:80px; height:5px; background:#E2E8F0; border-radius:999px; overflow:hidden;">
                            <div style="width:{p}%; height:100%; background:{bar_color};"></div>
                        </div>
                    </div>
                    """
                except Exception:
                    cell_content = cell_val
                    
            # Monospace hash formatting
            elif "0x" in cell_val or len(cell_val) > 28 and cell_val.isalnum():
                cell_content = f'<span style="font-family:monospace; color:#0F5BB6; font-size:0.8rem; background:#F1F5F9; padding:2px 6px; border-radius:4px;">{cell_val}</span>'
                
            td_cells += f'<td style="padding:14px 14px; font-size:0.84rem; color:#0F172A; border-bottom:1px solid #F1F5F9; vertical-align:middle;">{cell_content}</td>'
            
        rows_html += f'<tr style="transition:background 0.15s ease;" onmouseover="this.style.background=\'#F8FAFC\'" onmouseout="this.style.background=\'transparent\'">{td_cells}</tr>'
        
    pager_html = ""
    if pager:
        showing = pager.get("showing", "Showing all records")
        pages = pager.get("pages", [1, 2, 3])
        cur = pager.get("current", 1)
        p_btns = "".join(f'<span style="display:inline-block; width:26px; height:26px; line-height:26px; text-align:center; border-radius:6px; font-size:0.75rem; font-weight:600; margin:0 2px; background:{"#0F5BB6" if p == cur else "#FFFFFF"}; color:{"#FFFFFF" if p == cur else "#64748B"}; border:1px solid {"#0F5BB6" if p == cur else "#E2E8F0"}; cursor:pointer;">{p}</span>' for p in pages)
        pager_html = f"""
        <div style="display:flex; justify-content:space-between; align-items:center; padding:12px 14px 4px 14px; font-size:0.75rem; color:#64748B;">
            <div>{showing}</div>
            <div style="display:flex; align-items:center;">
                <span style="cursor:pointer; margin-right:6px;">{icon_svg("chevron_left", size=14, color="#64748B")}</span>
                {p_btns}
                <span style="cursor:pointer; margin-left:6px;">{icon_svg("chevron_right", size=14, color="#64748B")}</span>
            </div>
        </div>
        """

    raw = f"""
    <div style="width:100%; overflow-x:auto;">
        <table style="width:100%; border-collapse:collapse; text-align:left;">
            <thead>
                <tr>{th_cells}</tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        {pager_html}
    </div>
    """
    return clean_html(raw)


# ══════════════════════════════════════════════════════════════════════
#                     6. BUTTON PRIMITIVES
# ══════════════════════════════════════════════════════════════════════

def fc_button(
    label: str,
    kind: str = "primary",
    icon_name: Optional[str] = None,
    href: str = "#",
    class_name: str = ""
) -> str:
    """Render styled MedXChAln button (solid primary, outline secondary, or destructive outline)."""
    icon_part = f'<span style="margin-right:6px;">{icon_svg(icon_name, size=15)}</span>' if icon_name else ""
    raw = f"""
    <a href="{href}" class="fc-btn fc-btn-{kind} {class_name}" style="text-decoration:none; display:inline-flex; align-items:center; justify-content:center;">
        {icon_part}<span>{html.escape(label)}</span>
    </a>
    """
    return clean_html(raw)


def default_href_for_action(action_label: Optional[str]) -> str:
    """Infer target page route from header solid button action label."""
    if not action_label:
        return "/?page=Project+Overview"
    lbl = action_label.lower()
    if any(k in lbl for k in ["inference", "suggest", "predict", "diagnos"]):
        return "/?page=Data+Explorer"
    elif any(k in lbl for k in ["export", "report", "download", "figure"]):
        return "/?page=Research+Figures"
    elif any(k in lbl for k in ["aggregat", "round", "train", "model", "benchmark"]):
        return "/?page=Model+Comparison"
    elif any(k in lbl for k in ["node", "hosp", "facilit", "network"]):
        return "/?page=Hospital+Deep+Dive"
    elif any(k in lbl for k in ["verif", "ledger", "audit", "secur", "trace"]):
        return "/?page=Attack+vs.+Defense"
    elif any(k in lbl for k in ["intake", "patient", "anamnes"]):
        return "/?page=Data+Explorer"
    elif any(k in lbl for k in ["doc", "manual"]):
        return "/?page=Project+Overview"
    return "/?page=Project+Overview"


def fc_fab(icon_name: str = "plus", title: str = "Action", href: str = "/?page=Diagnosis+Support") -> str:
    """Render floating circular action button fixed at bottom-right."""
    target_attr = "" if href.startswith("javascript:") else 'target="_top"'
    raw = f"""
    <a href="{href}" {target_attr} class="fc-fab" title="{title}" style="text-decoration:none; display:flex; align-items:center; justify-content:center;">
        {icon_svg(icon_name, size=22, color="#FFFFFF", stroke_width=2.5)}
    </a>
    """
    return clean_html(raw)


# ══════════════════════════════════════════════════════════════════════
#                     7. HEADER COMPONENT (Per Page)
# ══════════════════════════════════════════════════════════════════════

def render_header(
    title: str,
    subtitle: Optional[str] = None,
    search_placeholder: Optional[str] = None,
    live_badge: Optional[str] = None,
    version_text: str = "v4.2.1-stable",
    last_sync: str = "2 mins ago",
    action_label: Optional[str] = None,
    action_icon: str = "zap",
    action_href: Optional[str] = None
) -> str:
    """
    Render top header row matching MedXChAln reference designs:
    LEFT: Title + subtitle or live status kicker
    RIGHT: Search form -> version/sync info -> icon button links -> solid action link
    """
    search_part = ""
    if search_placeholder:
        search_part = f"""
        <form action="/" method="GET" target="_top" style="display:inline-flex; align-items:center; margin:0; padding:0;">
            <div class="fc-header-search">
                <span style="color:#94A3B8; margin-right:8px;">{icon_svg("search", size=15)}</span>
                <input type="text" name="search" placeholder="{html.escape(search_placeholder)}" class="fc-search-input" />
            </div>
        </form>
        """
        
    badge_part = f'<span style="margin-left:10px;">{live_badge}</span>' if live_badge else ""
    sub_part = f'<p class="fc-header-subtitle">{html.escape(subtitle)}</p>' if subtitle else ""
    
    action_btn_part = ""
    if action_label:
        target_href = action_href or default_href_for_action(action_label)
        action_btn_part = f"""
        <a href="{target_href}" target="_top" class="fc-header-action-btn" style="text-decoration:none; display:inline-flex; align-items:center;">
            {icon_svg(action_icon, size=15, color="#FFFFFF", stroke_width=2.2)}
            <span style="margin-left:6px;">{html.escape(action_label)}</span>
        </a>
        """

    raw = f"""
    <div class="fc-header-bar">
        <!-- LEFT CLUSTER -->
        <div class="fc-header-left">
            <div style="display:flex; align-items:center;">
                <h1 class="fc-header-title">{html.escape(title)}</h1>
                {badge_part}
            </div>
            {sub_part}
        </div>
        
        <!-- RIGHT CLUSTER -->
        <div class="fc-header-right">
            {search_part}
            
            <!-- Live Model Status Kicker (Clickable to Model Management) -->
            <a href="/?page=Model+Comparison" target="_top" style="text-decoration:none;" title="View Global Model details">
                <div class="fc-header-meta" style="cursor:pointer;">
                    <span class="fc-header-meta-label">GLOBAL MODEL <strong style="color:#0F5BB6;">{version_text}</strong></span>
                    <span class="fc-header-meta-sync">Last sync: {last_sync}</span>
                </div>
            </a>
            
            <!-- Icon action links -->
            <div class="fc-header-icons">
                <a href="/?refresh=true" target="_top" class="fc-icon-btn" title="Refresh state & cache" style="text-decoration:none; display:inline-flex; align-items:center; justify-content:center;">
                    {icon_svg("refresh", size=16, color="#64748B")}
                </a>
                <a href="/?page=Hospital+Deep+Dive" target="_top" class="fc-icon-btn" title="Hospital Network" style="text-decoration:none; display:inline-flex; align-items:center; justify-content:center;">
                    {icon_svg("network", size=16, color="#64748B")}
                </a>
                <a href="/?page=Attack+vs.+Defense" target="_top" class="fc-icon-btn" title="Alerts & Traceability" style="text-decoration:none; display:inline-flex; align-items:center; justify-content:center;">
                    {icon_svg("bell_dot", size=16, color="#64748B")}
                </a>
            </div>
            
            <!-- Main solid button -->
            {action_btn_part}
        </div>
    </div>
    """
    return clean_html(raw)


# ══════════════════════════════════════════════════════════════════════
#                     8. SIDEBAR SHELL HELPERS
# ══════════════════════════════════════════════════════════════════════

def render_sidebar_header() -> str:
    """Render MedXChAln logo row and identity block for the top of the sidebar."""
    raw = f"""
    <a href="/?page=Project+Overview" target="_top" style="text-decoration:none; color:inherit;" title="Return to Project Overview">
        <div class="fc-sidebar-logo-row" style="cursor:pointer;">
            <div class="fc-logo-square">
                {icon_svg("heart", size=20, color="#FFFFFF", stroke_width=2.2)}
            </div>
            <div class="fc-wordmark">
                MedXChAIn <span style="font-size:0.65rem; color:#0F5BB6; background:#EFF6FF; padding:2px 6px; border-radius:4px; font-weight:700;">FedCare</span>
            </div>
        </div>
    </a>
    
    <a href="/?page=Project+Overview" target="_top" style="text-decoration:none; color:inherit;" title="Open Admin Profile">
        <div class="fc-identity-block" style="cursor:pointer;">
            <div class="fc-avatar-circle">
                {icon_svg("user", size=18, color="#0F5BB6", stroke_width=2.2)}
            </div>
            <div class="fc-identity-text">
                <div class="fc-identity-name">FedCare Admin</div>
                <div class="fc-identity-role">AI Researcher</div>
            </div>
        </div>
    </a>
    """
    return clean_html(raw)


def render_sidebar_footer(nodes_count: int = 6, sync_status: str = "Active") -> str:
    """Render bottom status card and settings/support rows for the sidebar."""
    raw = f"""
    <a href="/?page=Security+%26+Traceability" target="_top" style="text-decoration:none; color:inherit;" title="Traceability & Security Verification">
        <div class="fc-sidebar-status-card" style="cursor:pointer;">
            <div style="display:flex; align-items:center; gap:6px; margin-bottom:3px;">
                <span class="fc-status-pulse-dot"></span>
                <span style="font-size:0.7rem; font-weight:800; color:#0F5BB6; letter-spacing:0.05em; text-transform:uppercase;">
                    FEDERATED LEARNING {sync_status.upper()}
                </span>
            </div>
            <div style="font-size:0.72rem; color:#64748B; font-weight:500;">
                {nodes_count} nodes syncing • 99.9% Integrity
            </div>
        </div>
    </a>
    
    <div class="fc-sidebar-links">
        <a href="/?page=Privacy-Utility" target="_top" class="fc-sidebar-link-row" title="Configure Privacy & DP Parameters">
            {icon_svg("settings", size=16, color="#64748B")}
            <span>Settings</span>
        </a>
        <a href="/?page=Project+Overview" target="_top" class="fc-sidebar-link-row" title="Documentation & Architectural Guide">
            {icon_svg("help", size=16, color="#64748B")}
            <span>Support</span>
        </a>
    </div>
    """
    return clean_html(raw)


"""
app/ui_components.py
─────────────────────
Reusable Streamlit UI components and helpers.
Keeps the page files clean and focused on page logic.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import streamlit as st

# ── Sentiment badge renderer ──────────────────────────────────────────────────

SENTIMENT_CONFIG = {
    "Positive": {"emoji": "+", "color": "var(--pos)", "bg": "var(--pos-bg)"},
    "Negative": {"emoji": "-", "color": "var(--neg)", "bg": "var(--neg-bg)"},
    "Neutral":  {"emoji": "=", "color": "var(--neu)", "bg": "var(--neu-bg)"},
}


def sentiment_badge(sentiment: str) -> str:
    """Return an HTML sentiment badge string."""
    cfg = SENTIMENT_CONFIG.get(sentiment, {"emoji": "❓", "color": "#64748b", "bg": "#1e293b"})
    return (
        f'<span style="background:{cfg["bg"]};color:{cfg["color"]};'
        f'border:1px solid {cfg["color"]}55;border-radius:999px;'
        f'padding:3px 10px;font-weight:600;font-size:0.8rem;">'
        f'{cfg["emoji"]} {sentiment}</span>'
    )


def confidence_bar_html(confidence: float, sentiment: str) -> str:
    """Return an HTML confidence progress bar."""
    pct = int(confidence * 100)
    color = SENTIMENT_CONFIG.get(sentiment, {}).get("color", "var(--muted)")
    return (
        f'<div style="background:var(--line);border-radius:999px;height:7px;width:100%;">'
        f'<div style="width:{pct}%;background:{color};height:100%;border-radius:6px;'
        f'transition:width 0.4s ease;"></div></div>'
        f'<small style="color:var(--muted);">{pct}% confidence</small>'
    )


def metric_card(label: str, value: str, delta: Optional[str] = None,
                color: str = "var(--accent)") -> str:
    """Return an HTML metric card."""
    delta_html = ""
    if delta:
        delta_html = f'<div style="color:var(--muted);font-size:0.75rem;margin-top:4px;">{delta}</div>'
    return (
        f'<div style="background:var(--surface);border:1px solid var(--line);border-radius:8px;'
        f'padding:14px 16px;text-align:left;box-shadow:0 1px 2px rgba(16,42,67,.04);">'
        f'<div style="color:var(--muted);font-size:0.76rem;margin-bottom:6px;">{label}</div>'
        f'<div style="color:{color};font-size:1.65rem;font-weight:700;">{value}</div>'
        f'{delta_html}</div>'
    )


def section_header(title: str, subtitle: str = "") -> None:
    """Render a stylised section header."""
    st.markdown(
        f'<div style="margin:1.5rem 0 0.85rem;">'
        f'<h3 style="color:var(--ink);margin:0;font-weight:700;font-size:1.1rem;">{title}</h3>'
        + (f'<p style="color:var(--muted);margin:4px 0 0 0;font-size:0.86rem;">{subtitle}</p>'
           if subtitle else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def divider() -> None:
    st.markdown(
        '<hr style="border:0;border-top:1px solid var(--line);margin:2rem 0 1rem;">',
        unsafe_allow_html=True,
    )


def info_banner(message: str, kind: str = "info") -> None:
    """Render an info/warning/error banner."""
    colors = {
        "info":    ("var(--accent)",   "#e8f7f8"),
        "warning": ("var(--neu)",      "var(--neu-bg)"),
        "error":   ("var(--neg)",      "var(--neg-bg)"),
        "success": ("var(--pos)",      "var(--pos-bg)"),
    }
    border_color, bg_color = colors.get(kind, colors["info"])
    icons = {"info": "i", "warning": "!", "error": "x", "success": "ok"}
    icon = icons.get(kind, "i")
    st.markdown(
        f'<div style="background:{bg_color};border:1px solid {border_color}33;'
        f'border-left:3px solid {border_color};border-radius:6px;padding:11px 14px;margin:10px 0;">'
        f'<span style="color:{border_color};font-weight:700;margin-right:8px;">{icon}</span> '
        f'<span style="color:var(--ink);">{message}</span></div>',
        unsafe_allow_html=True,
    )


def prediction_card(aspect: str, sentiment: str, confidence: float,
                    probs: Dict[str, float], keywords: List[str]) -> None:
    """Render a full prediction result card."""
    cfg = SENTIMENT_CONFIG.get(sentiment, {"emoji": "❓", "color": "var(--muted)", "bg": "var(--bg)"})
    pct = int(confidence * 100)

    kw_html = ""
    if keywords:
        kw_chips = " ".join(
            f'<span style="background:var(--bg);color:var(--muted);border:1px solid var(--line);'
            f'border-radius:999px;padding:3px 9px;font-size:0.74rem;">{kw}</span>'
            for kw in keywords
        )
        kw_html = f'<div style="margin-top:8px;">{kw_chips}</div>'

    prob_bars = ""
    for label, prob in [("Positive", probs.get("Positive", 0)),
                         ("Negative", probs.get("Negative", 0)),
                         ("Neutral", probs.get("Neutral", 0))]:
        lc = SENTIMENT_CONFIG.get(label, {}).get("color", "var(--muted)")
        w = int(prob * 100)
        prob_bars += (
            f'<div style="display:flex;align-items:center;gap:8px;margin-top:4px;">'
            f'<span style="color:var(--muted);font-size:0.75rem;min-width:60px;">{label}</span>'
            f'<div style="flex:1;background:var(--bg);border-radius:4px;height:6px;">'
            f'<div style="width:{w}%;background:{lc};height:100%;border-radius:4px;"></div></div>'
            f'<span style="color:var(--muted);font-size:0.75rem;min-width:35px;">{prob:.1%}</span>'
            f'</div>'
        )

    st.markdown(
        f'<div style="background:var(--surface);border:1px solid var(--line);'
        f'border-left:3px solid {cfg["color"]};border-radius:8px;padding:16px;margin:8px 0;'
        f'box-shadow:0 1px 2px rgba(16,42,67,.04);">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<span style="color:var(--ink);font-weight:700;font-size:1rem;">{aspect}</span>'
        f'<span style="background:{cfg["bg"]};color:{cfg["color"]};border:1px solid {cfg["color"]}55;'
        f'border-radius:999px;padding:3px 10px;font-weight:600;font-size:0.8rem;">{cfg["emoji"]} {sentiment}</span>'
        f'</div>'
        f'<div style="margin-top:10px;">'
        f'<div style="background:var(--line);border-radius:999px;height:7px;">'
        f'<div style="width:{pct}%;background:{cfg["color"]};height:100%;border-radius:6px;"></div></div>'
        f'<small style="color:var(--muted);">{pct}% confidence</small>'
        f'</div>'
        f'<div style="margin-top:10px;">{prob_bars}</div>'
        f'{kw_html}'
        f'</div>',
        unsafe_allow_html=True,
    )

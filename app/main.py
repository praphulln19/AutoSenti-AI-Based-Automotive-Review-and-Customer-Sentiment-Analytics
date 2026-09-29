"""
app/main.py
\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
Streamlit entry point.

Run:
    streamlit run app/main.py
"""

from __future__ import annotations

import sys
import os

# Ensure the project root is on sys.path regardless of how Streamlit is invoked
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

# \u2500\u2500 Page config (MUST be the very first Streamlit call) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
st.set_page_config(
    page_title="AutoSenti \u2014 Vehicle Review Sentiment",
    page_icon=":material/directions_car:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "# AutoSenti\nAI-Based Automotive Review & Customer Sentiment Analytics",
    },
)

# \u2500\u2500 Global CSS \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# Unified design token system \u2014 ONE palette used everywhere:
#   --ink       : primary text / headings         (#102a43)
#   --muted     : secondary / caption text        (#627d98)
#   --line      : borders & dividers              (#d9e2ec)
#   --surface   : card / input background         (#ffffff)
#   --bg        : page background                 (#f4f7f9)
#   --accent    : primary brand teal              (#087f8c)
#   --accent-dk : darker teal for hover           (#05616b)
#   --pos       : positive sentiment green        (#047857)
#   --pos-bg    : positive sentiment bg           (#ecfdf5)
#   --neg       : negative sentiment red          (#b42318)
#   --neg-bg    : negative sentiment bg           (#fff1f0)
#   --neu       : neutral sentiment amber         (#946200)
#   --neu-bg    : neutral sentiment bg            (#fff8e1)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

:root {
  --ink:       #102a43;
  --muted:     #627d98;
  --line:      #d9e2ec;
  --surface:   #ffffff;
  --bg:        #f4f7f9;
  --accent:    #087f8c;
  --accent-dk: #05616b;
  --pos:       #047857;
  --pos-bg:    #ecfdf5;
  --neg:       #b42318;
  --neg-bg:    #fff1f0;
  --neu:       #946200;
  --neu-bg:    #fff8e1;
}

html, body, [class*="css"] { font-family:'IBM Plex Sans', sans-serif !important; }
.stApp { background:var(--bg); color:var(--ink); }
.block-container { max-width:1180px; padding:3.2rem 2rem 4rem; }
h1,h2,h3,h4,h5,h6 { color:var(--ink) !important; letter-spacing:0; }
p,li { color:var(--muted); }

/* Sidebar */
section[data-testid="stSidebar"] { background:var(--surface) !important; border-right:1px solid var(--line); }
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span { color:var(--muted) !important; }

/* Native Streamlit metrics */
div[data-testid="stMetric"] { background:var(--surface); border:1px solid var(--line); border-radius:8px; padding:14px 16px; box-shadow:0 1px 2px rgba(16,42,67,.04); }
div[data-testid="stMetricValue"] { color:var(--accent-dk) !important; }
div[data-testid="stMetricLabel"] { color:var(--muted) !important; }

/* Buttons */
.stButton > button { background:var(--accent) !important; color:#fff !important; border:1px solid var(--accent) !important; border-radius:6px !important; font-weight:600 !important; min-height:2.5rem; }
.stButton > button:hover { background:var(--accent-dk) !important; border-color:var(--accent-dk) !important; box-shadow:0 3px 10px rgba(8,127,140,.18) !important; }
.stButton > button:disabled { background:#bcccdc !important; border-color:#bcccdc !important; color:#fff !important; }

/* Text inputs */
div[data-testid="stTextInput"] input, div[data-testid="stTextArea"] textarea { background:var(--surface) !important; color:var(--ink) !important; border:1px solid #bcccdc !important; border-radius:6px !important; color-scheme:light; }
div[data-testid="stTextInput"] input:focus, div[data-testid="stTextArea"] textarea:focus { border-color:var(--accent) !important; box-shadow:0 0 0 2px rgba(8,127,140,.14) !important; }

/* Selectbox */
.stSelectbox > div > div { background:var(--surface) !important; border-color:#bcccdc !important; color:var(--ink) !important; border-radius:6px !important; }

/* Expander / DataFrame / Chart containers */
details { background:var(--surface); border:1px solid var(--line); border-radius:6px; }
div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:6px; overflow:hidden; }
div[data-testid="stPlotlyChart"] { background:var(--surface); border:1px solid var(--line); border-radius:8px; padding:8px 8px 0; }

/* Caption / small text */
.stCaption, small { color:var(--muted) !important; }

@media (max-width: 720px) { .block-container { padding:2.5rem 1rem 3rem; } h1 { font-size:1.8rem !important; } }
</style>
""", unsafe_allow_html=True)


# \u2500\u2500 Session state initialisation \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
def _init_state() -> None:
    defaults = {
        "engine": None,
        "model_loaded": False,
        "current_model_key": None,
        "current_checkpoint": None,
        "single_result": None,
        # Active page: "analyze" | "training"
        "_active_page": "analyze",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()


# \u2500\u2500 Lazy engine loader \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
@st.cache_resource(show_spinner=False)
def get_engine():
    from src.inference.inference_engine import InferenceEngine
    return InferenceEngine()


# \u2500\u2500 Import pages \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
from app.pages import page_home, page_training_metrics  # noqa: E402

# Page registry used for nav buttons
_PAGES = {
    "analyze":  {"label": "Analyze vehicle reviews",       "icon": "analytics"},
    "training": {"label": "Training & Evaluation Metrics", "icon": "monitoring"},
}


# \u2500\u2500 Sidebar \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# NOTE: st.navigation is NOT used so that we can write to st.sidebar freely.
# Active page is tracked in session_state and switched via buttons below.
with st.sidebar:

    # Brand header
    st.markdown("""
    <div style="padding:12px 0 18px;">
        <div style="display:flex;align-items:center;gap:10px;">
            <div style="width:34px;height:34px;background:var(--accent);color:#fff;border-radius:6px;
                        display:flex;align-items:center;justify-content:center;font-weight:700;">AS</div>
            <div>
                <div style="color:var(--ink);font-size:1.05rem;font-weight:700;line-height:1.1;">AutoSenti</div>
                <div style="color:var(--muted);font-size:0.72rem;margin-top:3px;">Vehicle review insights</div>
            </div>
        </div>
        <p style="color:var(--muted);font-size:0.78rem;margin:16px 0 0;line-height:1.45;">
            Analyze sentiment for individual vehicle aspects.
        </p>
    </div>
    <hr style="border:0;border-top:1px solid var(--line);margin:0 0 14px;">
    """, unsafe_allow_html=True)

    # \u2500\u2500 Navigation buttons \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    st.markdown(
        '<p style="color:var(--muted);font-size:0.72rem;font-weight:700;'
        'margin-bottom:6px;letter-spacing:0.08em;">NAVIGATION</p>',
        unsafe_allow_html=True,
    )
    active = st.session_state["_active_page"]
    for page_key, page_info in _PAGES.items():
        clicked = st.button(
            f":material/{page_info['icon']}: {page_info['label']}",
            key=f"_nav_{page_key}",
            use_container_width=True,
            type="primary" if page_key == active else "secondary",
        )
        if clicked and page_key != active:
            st.session_state["_active_page"] = page_key
            st.rerun()

    st.markdown(
        '<hr style="border:0;border-top:1px solid var(--line);margin:14px 0;">',
        unsafe_allow_html=True,
    )

    # \u2500\u2500 Model loader \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    st.markdown(
        '<p style="color:var(--muted);font-size:0.72rem;font-weight:700;'
        'margin-bottom:6px;letter-spacing:0.08em;">MODEL</p>',
        unsafe_allow_html=True,
    )

    from src.config_loader import load_config
    cfg = load_config()
    model_options = {v["display_name"]: k for k, v in cfg["models"].items()}

    selected_display = st.selectbox(
        "Model",
        list(model_options.keys()),
        label_visibility="collapsed",
    )
    selected_model_key = model_options[selected_display]

    if st.button("Load model", use_container_width=True):
        engine = get_engine()
        with st.spinner(f"Loading {selected_display}\u2026"):
            try:
                engine.load_model(selected_model_key)
                ckpt = engine._current_checkpoint
                st.session_state["model_loaded"] = True
                st.session_state["current_model_key"] = selected_model_key
                st.session_state["current_checkpoint"] = ckpt
                st.session_state["engine"] = engine
                if ckpt:
                    import os as _os
                    st.success(
                        f"{selected_display} ready\n\n"
                        f"Checkpoint: `{_os.path.basename(ckpt)}`"
                    )
                else:
                    st.warning("Loaded pretrained weights; fine-tuning is not complete yet.")
            except Exception as exc:
                st.error(f"Failed to load: {exc}")

    # Status indicator
    st.markdown('<div style="margin-top:8px;">', unsafe_allow_html=True)
    if st.session_state.get("model_loaded"):
        mk      = st.session_state.get("current_model_key", "")
        display = cfg["models"].get(mk, {}).get("display_name", mk)
        ckpt    = st.session_state.get("current_checkpoint")
        ckpt_label = (
            f"<br><span style='color:var(--pos);font-size:0.65rem;'>"
            f"\U0001f4c1 {os.path.basename(ckpt)}</span>"
        ) if ckpt else ""
        st.markdown(
            f'<div style="background:var(--pos-bg);border:1px solid #a7f3d0;'
            f'border-radius:6px;padding:10px;text-align:center;">'
            f'<span style="color:var(--pos);font-size:0.8rem;font-weight:600;">{display}</span>'
            f'{ckpt_label}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="background:var(--neg-bg);border:1px solid #fecaca;'
            'border-radius:6px;padding:10px;text-align:center;">'
            '<span style="color:var(--neg);font-size:0.8rem;">No model loaded</span><br>'
            '<span style="color:var(--neg);font-size:0.68rem;opacity:0.8;">'
            'Select a model to begin</span></div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '<hr style="border:0;border-top:1px solid var(--line);margin:20px 0 10px;">',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:var(--muted);font-size:0.68rem;text-align:center;margin:0;">'
        'Aspect-based sentiment analysis</p>',
        unsafe_allow_html=True,
    )


# \u2500\u2500 Render active page \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# Pages render directly \u2014 no st.navigation() call to avoid the sidebar
# ownership conflict introduced in Streamlit \u2265 1.44.
if st.session_state["_active_page"] == "training":
    page_training_metrics.render()
else:
    page_home.render()

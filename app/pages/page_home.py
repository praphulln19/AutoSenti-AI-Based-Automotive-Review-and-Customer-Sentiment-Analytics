"""
app/pages/page_home.py
───────────────────────
Vehicle review aspect sentiment page.

Responsibilities
----------------
- Accept a review from the user (text area or example buttons).
- Run ABSA through the inference engine.
- Display structured prediction cards with confidence and probability bars.
- Show a confidence bar chart.
"""

from __future__ import annotations

import streamlit as st

from app.ui_components import (
    section_header, divider, info_banner, prediction_card, metric_card
)
from src.data.validator import validate_review_text


# ── Example reviews ───────────────────────────────────────────────────────────
EXAMPLES = [
    "The battery range is excellent, but the charging time is too long.",
    "The mileage is impressive and the engine is very responsive, though the service at the dealership was disappointing.",
    "Comfortable seats and a smooth ride, but the infotainment system keeps lagging.",
    "Great value for money — this car offers safety features you'd normally only find in luxury vehicles.",
    "The vehicle looks stunning and the fuel efficiency is remarkable. However, the rear legroom is too cramped for a family.",
]


def render():
    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="padding:0 0 0.5rem;">
        <div style="color:var(--accent);font-size:0.74rem;font-weight:700;letter-spacing:0.08em;">
            VEHICLE REVIEW INSIGHTS
        </div>
        <h1 style="color:var(--ink);font-weight:700;font-size:2.15rem;line-height:1.15;margin:0.35rem 0 0;">
            Analyze vehicle reviews
        </h1>
        <p style="color:var(--muted);margin:0.55rem 0 0;font-size:0.98rem;">
            Turn a customer review into clear sentiment signals for each vehicle aspect.
        </p>
    </div>
    """, unsafe_allow_html=True)

    divider()

    # ── Model check ──────────────────────────────────────────────────────────
    if not st.session_state.get("model_loaded"):
        info_banner(
            "No model is loaded. Select a model in the sidebar before analyzing reviews.",
            kind="warning",
        )

    # ── Input section ────────────────────────────────────────────────────────
    # Apply any pending example text BEFORE the widget is instantiated
    if "_review_input_pending" in st.session_state:
        st.session_state["single_review_input"] = st.session_state.pop("_review_input_pending")

    col_input, col_examples = st.columns([2.35, 1], gap="large")

    with col_input:
        section_header("Vehicle review", "Type or paste a customer review below")
        review_text = st.text_area(
            "Review",
            height=140,
            placeholder="e.g. The battery range is excellent, but the charging time is too long.",
            label_visibility="collapsed",
            key="single_review_input",
        )
        st.caption(f"{len(review_text)} / 1000 characters")

    with col_examples:
        section_header("Try an example", "Start with a sample review")
        example_options = ["Choose an example"] + EXAMPLES
        selected_example = st.selectbox(
            "Example review",
            example_options,
            format_func=lambda value: value if len(value) < 48 else value[:45] + "...",
            label_visibility="collapsed",
        )
        if st.button("Use example", width="stretch", icon=":material/arrow_downward:"):
            if selected_example != "Choose an example":
                st.session_state["_review_input_pending"] = selected_example
                st.session_state["single_result"] = None
                st.rerun()

    # ── Analyse button ───────────────────────────────────────────────────────
    col_btn, col_clear = st.columns([1, 1], gap="small")
    with col_btn:
        analyse_clicked = st.button(
            "Analyze review",
            width="stretch",
            disabled=not st.session_state.get("model_loaded"),
            icon=":material/analytics:",
        )
    with col_clear:
        if st.button("Clear", width="stretch", icon=":material/refresh:"):
            st.session_state["single_review_input"] = ""
            st.session_state["single_result"] = None
            st.rerun()

    # ── Analysis ─────────────────────────────────────────────────────────────
    if analyse_clicked:
        text = st.session_state.get("single_review_input", "").strip()

        # Validate
        val = validate_review_text(text)
        if not val.is_valid:
            for err in val.errors:
                info_banner(err, kind="error")
            return

        for warn in val.warnings:
            info_banner(warn, kind="warning")

        engine = st.session_state.get("engine")
        if engine is None:
            info_banner("Engine not initialised. Please reload the model.", kind="error")
            return

        with st.spinner("Analyzing review..."):
            try:
                result = engine.analyse_review(text)
                st.session_state["single_result"] = result
            except Exception as exc:
                info_banner(f"Analysis failed: {exc}", kind="error")
                return

    # ── Display results ───────────────────────────────────────────────────────
    result = st.session_state.get("single_result")
    if result is None:
        return

    divider()
    section_header("Aspect sentiment results", "Signals detected in the submitted review")

    if result.no_aspects_detected:
        info_banner(
            "No recognisable automotive aspects were detected in this review. "
            "Try rephrasing or including more specific automotive terms.",
            kind="warning",
        )
        return

    # Summary metrics row
    preds = result.predictions
    sentiments = [p.sentiment for p in preds]
    pos_count = sentiments.count("Positive")
    neg_count = sentiments.count("Negative")
    neu_count = sentiments.count("Neutral")
    avg_conf = sum(p.confidence for p in preds) / len(preds) if preds else 0

    cols = st.columns(4)
    metric_data = [
        ("Aspects Detected", str(len(preds)), None, "var(--accent)"),
        ("Positive", str(pos_count), None, "var(--pos)"),
        ("Negative", str(neg_count), None, "var(--neg)"),
        ("Avg Confidence", f"{avg_conf:.1%}", None, "var(--neu)"),
    ]
    for col, (label, value, delta, color) in zip(cols, metric_data):
        with col:
            st.markdown(metric_card(label, value, delta, color), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

    # Prediction cards
    section_header("Aspect sentiment breakdown")
    cols_per_row = 2
    rows = [preds[i:i+cols_per_row] for i in range(0, len(preds), cols_per_row)]
    for row in rows:
        cols = st.columns(len(row))
        for col, pred in zip(cols, row):
            with col:
                prediction_card(
                    aspect=pred.aspect,
                    sentiment=pred.sentiment,
                    confidence=pred.confidence,
                    probs=pred.probabilities,
                    keywords=pred.triggered_keywords,
                )

    # Confidence chart
    divider()
    section_header("Confidence overview", "How strongly the model supports each prediction")
    from src.analytics.visualizer import plot_confidence_bar
    conf_df = result.as_dataframe
    fig = plot_confidence_bar(conf_df)
    st.plotly_chart(fig, width="stretch")

    # Raw results table
    with st.expander("View raw results", icon=":material/table_chart:"):
        st.dataframe(conf_df, width="stretch")

"""
Sedentary Health Risk Dashboard
================================
PhD Research Application — Yogesh Kumar Singh
IIT Kharagpur, Department of Computer Science & Engineering
Research: "Modelling Physiological and Cognitive Health for Sedentary Lifestyle using Data Analytics"
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sedentary Health Risk Dashboard | IIT Kharagpur",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "Sedentary Health Risk Dashboard — PhD Research, IIT Kharagpur",
    },
)

# ─────────────────────────────────────────────────────────────
# RESEARCH CONSTANTS (exact data from the study)
# ─────────────────────────────────────────────────────────────

# Study demographics
STUDY_N = 60
STUDY_MALE = 43
STUDY_FEMALE = 17
STUDY_AGE_RANGE = "24–40"

# Study means
MEAN_SITTING = 9.6
SD_SITTING = 2.1
MEAN_SCREEN = 3.4
SD_SCREEN = 1.2
MEAN_SLEEP = 7.2
SD_SLEEP = 0.8
MEAN_STRESS = 6.5
SD_STRESS = 1.7
MEAN_MENTAL_FATIGUE = 6.9
SD_MENTAL_FATIGUE = 1.5
MEAN_ACTIVITY = 2.1
SD_ACTIVITY = 1.4

# Sedentary categories
SED_CATEGORIES = ["Low (<8h/day)", "Moderate (8–10h/day)", "High (>10h/day)"]
SED_COUNTS = [12, 23, 25]
SED_PERCENTS = [20.0, 38.3, 41.7]
MENTAL_FATIGUE_BY_CAT = [4.8, 6.4, 8.1]

# Musculoskeletal pain prevalence (%)
PAIN_SITES = ["Lower Back", "Neck", "Shoulder", "Wrist"]
PAIN_PREV = [58, 52, 43, 27]

# Pearson correlations
CORR_VARS = [
    "Sitting Time",
    "Mental Fatigue",
    "Stress Score",
    "Concentration",
    "Lower Back Pain",
    "Physical Activity",
    "Sleep Quality",
]
# Correlation matrix (symmetric, diagonal = 1.0)
# Values derived from reported Pearson correlations
CORR_MATRIX = np.array([
    # Sit   MF     Stress  Conc   LBP    PA     Sleep
    [1.00,  0.71,  0.64,   0.68,  0.73, -0.38, -0.29],  # Sitting Time
    [0.71,  1.00,  0.57,   0.61,  0.52, -0.49, -0.42],  # Mental Fatigue
    [0.64,  0.57,  1.00,   0.55,  0.48, -0.41, -0.56],  # Stress Score
    [0.68,  0.61,  0.55,   1.00,  0.50, -0.37, -0.33],  # Concentration
    [0.73,  0.52,  0.48,   0.50,  1.00, -0.31, -0.25],  # Lower Back Pain
    [-0.38,-0.49, -0.41,  -0.37, -0.31,  1.00,  0.35],  # Physical Activity
    [-0.29,-0.42, -0.56,  -0.33, -0.25,  0.35,  1.00],  # Sleep Quality
])

# Random Forest variable importance (%)
RF_VARS = [
    "Total Sitting Time",
    "Stress Score",
    "Sleep Quality",
    "Physical Activity",
    "Screen Time",
    "BMI",
]
RF_IMPORTANCE = [27.8, 22.4, 15.6, 14.9, 10.3, 9.0]

# Regression coefficients (standardised β)
REG_VARS = ["Sitting Time", "Stress Score", "Physical Activity", "Sleep Duration", "Screen Time"]
REG_BETAS = [0.42, 0.38, -0.21, -0.17, 0.14]
REG_R2 = 0.62

# SRI distribution
SRI_LABELS = ["Low Risk", "Moderate Risk", "High Risk"]
SRI_COUNTS_DIST = [14, 27, 19]
SRI_PERCENTS_DIST = [23.3, 45.0, 31.7]

# Colour palette
C_LOW = "#00c853"
C_MOD = "#ff9800"
C_HIGH = "#f44336"
C_BLUE = "#1565c0"
C_CARD_BG = "#1e2130"
C_TEXT = "#e0e0e0"
C_ACCENT = "#42a5f5"

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Main container padding */
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    /* Metric card */
    .metric-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 1.1rem 1.4rem;
        margin-bottom: 0.8rem;
        border: 1px solid #2a2f45;
    }
    .metric-card .label {
        font-size: 0.78rem;
        color: #9e9e9e;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.25rem;
    }
    .metric-card .value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #e0e0e0;
        line-height: 1.1;
    }
    .metric-card .sub {
        font-size: 0.8rem;
        color: #757575;
        margin-top: 0.2rem;
    }

    /* Risk badge */
    .risk-badge {
        display: inline-block;
        padding: 0.45rem 1.4rem;
        border-radius: 50px;
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        margin: 0.4rem 0;
    }
    .risk-low    { background: rgba(0,200,83,0.18);  color: #00c853; border: 1.5px solid #00c853; }
    .risk-mod    { background: rgba(255,152,0,0.18); color: #ff9800; border: 1.5px solid #ff9800; }
    .risk-high   { background: rgba(244,67,54,0.18); color: #f44336; border: 1.5px solid #f44336; }

    /* Section header */
    .section-header {
        font-size: 0.72rem;
        font-weight: 600;
        color: #9e9e9e;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid #2a2f45;
        margin-bottom: 0.9rem;
    }

    /* Info card */
    .info-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border: 1px solid #2a2f45;
        margin-bottom: 1rem;
    }
    .info-card h4 {
        color: #42a5f5;
        font-size: 0.95rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    .info-card p {
        color: #bdbdbd;
        font-size: 0.88rem;
        line-height: 1.55;
        margin: 0;
    }

    /* Finding highlight */
    .finding-row {
        display: flex;
        align-items: center;
        gap: 0.9rem;
        background: #1e2130;
        border-radius: 10px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.55rem;
        border-left: 3px solid #1565c0;
    }
    .finding-num {
        font-size: 1.4rem;
        font-weight: 800;
        color: #1565c0;
        min-width: 3.5rem;
    }
    .finding-text { font-size: 0.88rem; color: #bdbdbd; line-height: 1.4; }

    /* Sidebar branding */
    .sidebar-brand {
        background: linear-gradient(135deg, #1565c0 0%, #0d47a1 100%);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    .sidebar-brand .title { font-size: 0.88rem; font-weight: 700; color: #fff; }
    .sidebar-brand .sub   { font-size: 0.74rem; color: #90caf9; margin-top: 0.2rem; }

    /* Divider */
    .custom-divider { border: none; border-top: 1px solid #2a2f45; margin: 1.2rem 0; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0e1117; }
    ::-webkit-scrollbar-thumb { background: #2a2f45; border-radius: 3px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def compute_sri(sitting, screen, sleep, stress, activity, bmi):
    """
    Compute the Sedentary Risk Index (0–100) using Random Forest
    variable importances as weights.

    Weights:
        sitting    0.278  (higher = worse)
        stress     0.224  (higher = worse)
        sleep      0.156  (lower = worse → invert as 10-sleep)
        activity   0.149  (lower = worse → invert as 7-activity)
        screen     0.103  (higher = worse)
        bmi        0.090  (deviation from 18.5 = worse)
    """
    raw = (
        sitting * 0.278
        + stress * 0.224
        + (10.0 - sleep) * 0.156
        + (7.0 - activity) * 0.149
        + screen * 0.103
        + max(bmi - 18.5, 0.0) * 0.09
    )
    # Calibrated min/max for the slider ranges:
    # min: sitting=4, stress=1, sleep=10, activity=7, screen=1, bmi=18.5
    raw_min = 4*0.278 + 1*0.224 + 0*0.156 + 0*0.149 + 1*0.103 + 0*0.09
    # max: sitting=16, stress=10, sleep=4, activity=0, screen=12, bmi=40
    raw_max = 16*0.278 + 10*0.224 + 6*0.156 + 7*0.149 + 12*0.103 + 21.5*0.09
    sri = (raw - raw_min) / (raw_max - raw_min) * 100.0
    return float(np.clip(sri, 0, 100))


def sri_category(sri):
    if sri < 33:
        return "Low Risk", C_LOW, "risk-low"
    elif sri < 67:
        return "Moderate Risk", C_MOD, "risk-mod"
    else:
        return "High Risk", C_HIGH, "risk-high"


def predict_mental_fatigue(sitting, stress, activity, sleep, screen):
    """
    Predicted mental fatigue using standardised regression coefficients
    centred on study means. Returns a 1–10 score.
    """
    delta = (
        0.42 * (sitting - MEAN_SITTING) / SD_SITTING
        + 0.38 * (stress - MEAN_STRESS) / SD_STRESS
        - 0.21 * (activity - MEAN_ACTIVITY) / SD_ACTIVITY
        - 0.17 * (sleep - MEAN_SLEEP) / SD_SLEEP
        + 0.14 * (screen - MEAN_SCREEN) / SD_SCREEN
    )
    # Scale delta: max |delta| ≈ 3 std → maps to ±2.5 fatigue units
    predicted = MEAN_MENTAL_FATIGUE + delta * (SD_MENTAL_FATIGUE / 1.0)
    return float(np.clip(predicted, 1.0, 10.0))


def gauge_chart(sri, color):
    """Plotly gauge for SRI score."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=sri,
            number={"font": {"size": 48, "color": color}, "suffix": ""},
            title={"text": "Sedentary Risk Index", "font": {"size": 14, "color": "#9e9e9e"}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#555",
                    "tickvals": [0, 33, 67, 100],
                    "ticktext": ["0", "33", "67", "100"],
                    "tickfont": {"color": "#9e9e9e", "size": 11},
                },
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "#1e2130",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 33],  "color": "rgba(0,200,83,0.12)"},
                    {"range": [33, 67], "color": "rgba(255,152,0,0.12)"},
                    {"range": [67, 100],"color": "rgba(244,67,54,0.12)"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 3},
                    "thickness": 0.85,
                    "value": sri,
                },
            },
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=30, b=10),
        height=260,
        font=dict(color="#e0e0e0"),
    )
    return fig


def metric_card_html(label, value, sub="", color=C_ACCENT):
    return f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value" style="color:{color};">{value}</div>
        <div class="sub">{sub}</div>
    </div>
    """


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="title">🏛️ IIT Kharagpur</div>
            <div class="sub">Rekhi Centre for Happiness Science</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        ["🏥 Risk Calculator", "📊 Research Insights", "🔬 About the Research"],
        label_visibility="collapsed",
    )

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style='font-size:0.78rem; color:#757575; line-height:1.6;'>
            <b style='color:#9e9e9e;'>Researcher</b><br>
            Yogesh Kumar Singh<br>
            PhD Scholar, CSE<br><br>
            <b style='color:#9e9e9e;'>Study Cohort</b><br>
            60 participants<br>
            Age 24–40 years<br>
            43M / 17F<br><br>
            <b style='color:#9e9e9e;'>Model Accuracy</b><br>
            R² = 0.62 (mental fatigue)<br>
            RF variable importance validated
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.7rem; color:#424242; text-align:center;'>For research & educational use only</div>",
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════
# PAGE 1 — RISK CALCULATOR
# ═════════════════════════════════════════════════════════════
if page == "🏥 Risk Calculator":

    st.markdown("## Sedentary Health Risk Calculator")
    st.markdown(
        "<span style='color:#9e9e9e; font-size:0.92rem;'>Based on IIT Kharagpur PhD Research — "
        "enter your daily habits to compute your personalised Sedentary Risk Index (SRI)</span>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    col_inputs, col_results = st.columns([1, 1.05], gap="large")

    # ── Inputs ──────────────────────────────────────────────
    with col_inputs:
        st.markdown("<div class='section-header'>Health Parameters</div>", unsafe_allow_html=True)

        sitting = st.slider(
            "🪑 Daily Sitting Time (hours)",
            min_value=4.0, max_value=16.0, value=9.0, step=0.5,
            help="Total hours spent sitting across work, commute, and leisure",
        )
        screen = st.slider(
            "🖥️ Daily Screen Time (hours, leisure)",
            min_value=1.0, max_value=12.0, value=3.0, step=0.5,
            help="Recreational screen time excluding work (phone, TV, gaming)",
        )
        sleep = st.slider(
            "😴 Sleep Duration (hours)",
            min_value=4.0, max_value=10.0, value=7.0, step=0.25,
            help="Average nightly sleep duration",
        )
        stress = st.slider(
            "😰 Stress Level (1 = calm, 10 = severe)",
            min_value=1, max_value=10, value=6,
            help="Self-reported perceived stress on a 1–10 scale",
        )
        activity = st.slider(
            "🏃 Physical Activity (days/week)",
            min_value=0, max_value=7, value=2,
            help="Days per week with at least 30 min of moderate-intensity exercise",
        )
        bmi = st.slider(
            "⚖️ BMI (kg/m²)",
            min_value=15.0, max_value=40.0, value=24.0, step=0.5,
            help="Body Mass Index — weight(kg) ÷ height(m)²",
        )

        st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:0.76rem; color:#616161; line-height:1.6;'>"
            "<b style='color:#757575;'>Study averages for reference</b><br>"
            f"Sitting: {MEAN_SITTING}±{SD_SITTING}h &nbsp;|&nbsp; "
            f"Screen: {MEAN_SCREEN}±{SD_SCREEN}h<br>"
            f"Sleep: {MEAN_SLEEP}±{SD_SLEEP}h &nbsp;|&nbsp; "
            f"Stress: {MEAN_STRESS}±{SD_STRESS}<br>"
            f"Activity: {MEAN_ACTIVITY}±{SD_ACTIVITY} days/wk"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Results ─────────────────────────────────────────────
    with col_results:
        sri = compute_sri(sitting, screen, sleep, stress, activity, bmi)
        cat_label, cat_color, cat_css = sri_category(sri)
        mf_pred = predict_mental_fatigue(sitting, stress, activity, sleep, screen)

        st.markdown("<div class='section-header'>Risk Assessment Results</div>", unsafe_allow_html=True)

        # Risk badge
        st.markdown(
            f"<div style='text-align:center; margin-bottom:0.5rem;'>"
            f"<span class='risk-badge {cat_css}'>{cat_label}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Gauge
        st.plotly_chart(gauge_chart(sri, cat_color), use_container_width=True, config={"displayModeBar": False})

        # Interpretation
        if sri < 33:
            interp = (
                "Your current lifestyle shows <b style='color:#00c853;'>low sedentary risk</b>. "
                "You are below the study mean for sitting time and maintain adequate physical activity. "
                "Continue your current habits and monitor regularly."
            )
        elif sri < 67:
            interp = (
                "Your lifestyle indicates <b style='color:#ff9800;'>moderate sedentary risk</b>. "
                "This aligns with the most common risk profile in the IIT KGP study (45% of participants). "
                "Consider increasing daily movement and improving sleep quality to reduce risk."
            )
        else:
            interp = (
                "Your profile reflects <b style='color:#f44336;'>high sedentary risk</b>. "
                "Participants in this category showed significantly elevated mental fatigue (avg 8.1/10) "
                "and higher musculoskeletal pain prevalence. Immediate lifestyle interventions are recommended."
            )

        st.markdown(
            f"<div class='info-card' style='margin-top:0;'>"
            f"<p>{interp}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Metric cards
        st.markdown("<div class='section-header' style='margin-top:0.5rem;'>Predicted Health Indicators</div>", unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        mf_color = C_LOW if mf_pred < 5 else (C_MOD if mf_pred < 7.5 else C_HIGH)
        stress_impact = min(100, round(stress / 10 * 100 + (sitting - 8) * 2.5))
        stress_impact = max(0, stress_impact)
        stress_color = C_LOW if stress_impact < 40 else (C_MOD if stress_impact < 70 else C_HIGH)
        ph_risk = min(100, round(sri * 0.85 + (10 - activity * 10) * 0.15))
        ph_color = C_LOW if ph_risk < 33 else (C_MOD if ph_risk < 67 else C_HIGH)

        with m1:
            st.markdown(
                metric_card_html(
                    "Predicted Mental Fatigue",
                    f"{mf_pred:.1f}/10",
                    f"Study mean: {MEAN_MENTAL_FATIGUE}/10",
                    mf_color,
                ),
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                metric_card_html(
                    "Stress Impact Score",
                    f"{stress_impact}%",
                    "Composite stress burden",
                    stress_color,
                ),
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                metric_card_html(
                    "Physical Health Risk",
                    f"{ph_risk:.0f}%",
                    "Musculoskeletal risk proxy",
                    ph_color,
                ),
                unsafe_allow_html=True,
            )

        # SRI breakdown bar
        st.markdown("<div class='section-header' style='margin-top:0.2rem;'>SRI Score Breakdown</div>", unsafe_allow_html=True)

        contrib_labels = ["Sitting", "Stress", "Poor Sleep", "Inactivity", "Screen Time", "BMI"]
        contrib_values = [
            round(sitting * 0.278, 2),
            round(stress * 0.224, 2),
            round((10 - sleep) * 0.156, 2),
            round((7 - activity) * 0.149, 2),
            round(screen * 0.103, 2),
            round(max(bmi - 18.5, 0) * 0.09, 2),
        ]
        contrib_colors = [C_HIGH, C_MOD, C_MOD, C_ACCENT, "#ab47bc", "#26a69a"]

        fig_contrib = go.Figure(
            go.Bar(
                x=contrib_labels,
                y=contrib_values,
                marker_color=contrib_colors,
                text=[f"{v:.2f}" for v in contrib_values],
                textposition="outside",
                textfont=dict(size=10, color="#e0e0e0"),
            )
        )
        fig_contrib.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=35),
            height=175,
            yaxis=dict(showgrid=True, gridcolor="#2a2f45", title="Weighted Score", title_font=dict(size=10)),
            xaxis=dict(tickfont=dict(size=10)),
            showlegend=False,
        )
        st.plotly_chart(fig_contrib, use_container_width=True, config={"displayModeBar": False})


# ═════════════════════════════════════════════════════════════
# PAGE 2 — RESEARCH INSIGHTS
# ═════════════════════════════════════════════════════════════
elif page == "📊 Research Insights":

    st.markdown("## Research Insights")
    st.markdown(
        "<span style='color:#9e9e9e; font-size:0.92rem;'>Statistical findings from the IIT Kharagpur cohort study "
        f"(N={STUDY_N}, age {STUDY_AGE_RANGE} years)</span>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # ── ROW 1 ────────────────────────────────────────────────
    r1c1, r1c2 = st.columns(2, gap="medium")

    with r1c1:
        # Variable importance bar chart
        sorted_pairs = sorted(zip(RF_IMPORTANCE, RF_VARS), reverse=True)
        sorted_imp, sorted_vars = zip(*sorted_pairs)

        # Gradient from blue (low) to red (high importance)
        imp_colors = [
            f"rgb({int(21+233*(v/max(sorted_imp)))}, {int(101-101*(v/max(sorted_imp)))}, {int(192-192*(v/max(sorted_imp)))})"
            for v in sorted_imp
        ]

        fig_rf = go.Figure(
            go.Bar(
                y=list(sorted_vars),
                x=list(sorted_imp),
                orientation="h",
                marker=dict(color=list(imp_colors)),
                text=[f"{v}%" for v in sorted_imp],
                textposition="outside",
                textfont=dict(size=11, color="#e0e0e0"),
            )
        )
        fig_rf.update_layout(
            title=dict(text="Variable Importance in Predicting Health Outcomes", font=dict(size=13), x=0),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=50, t=45, b=40),
            height=310,
            xaxis=dict(
                title="Importance (%)",
                range=[0, 35],
                gridcolor="#2a2f45",
                title_font=dict(size=11),
            ),
            yaxis=dict(tickfont=dict(size=11)),
            showlegend=False,
        )
        st.plotly_chart(fig_rf, use_container_width=True, config={"displayModeBar": False})

    with r1c2:
        # Mental fatigue by sedentary category
        cat_colors = [C_LOW, C_MOD, C_HIGH]
        fig_mf = go.Figure(
            go.Bar(
                x=["Low\n(<8h/day)", "Moderate\n(8–10h/day)", "High\n(>10h/day)"],
                y=MENTAL_FATIGUE_BY_CAT,
                marker_color=cat_colors,
                text=[f"{v}/10" for v in MENTAL_FATIGUE_BY_CAT],
                textposition="outside",
                textfont=dict(size=13, color="#e0e0e0"),
                width=0.5,
            )
        )
        fig_mf.add_hline(
            y=MEAN_MENTAL_FATIGUE,
            line_dash="dot",
            line_color="#9e9e9e",
            annotation_text=f"Study mean: {MEAN_MENTAL_FATIGUE}",
            annotation_font=dict(size=10, color="#9e9e9e"),
            annotation_position="bottom right",
        )
        fig_mf.update_layout(
            title=dict(text="Mental Fatigue Score by Sedentary Category", font=dict(size=13), x=0),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=20, t=45, b=40),
            height=310,
            yaxis=dict(
                title="Mental Fatigue (1–10)",
                range=[0, 10.5],
                gridcolor="#2a2f45",
                title_font=dict(size=11),
            ),
            xaxis=dict(tickfont=dict(size=11)),
            showlegend=False,
        )
        st.plotly_chart(fig_mf, use_container_width=True, config={"displayModeBar": False})

    # ── ROW 2 ────────────────────────────────────────────────
    r2c1, r2c2 = st.columns(2, gap="medium")

    with r2c1:
        # Correlation heatmap
        corr_labels = ["Sitting\nTime", "Mental\nFatigue", "Stress\nScore",
                       "Conc.\nProblems", "Lower\nBack Pain", "Physical\nActivity", "Sleep\nQuality"]
        fig_hm = go.Figure(
            go.Heatmap(
                z=CORR_MATRIX,
                x=corr_labels,
                y=corr_labels,
                colorscale=[
                    [0.0,  "#1565c0"],
                    [0.35, "#283593"],
                    [0.5,  "#1e2130"],
                    [0.65, "#b71c1c"],
                    [1.0,  "#f44336"],
                ],
                zmid=0,
                zmin=-1,
                zmax=1,
                text=np.round(CORR_MATRIX, 2),
                texttemplate="%{text}",
                textfont=dict(size=9, color="#e0e0e0"),
                colorbar=dict(
                    tickfont=dict(size=9, color="#9e9e9e"),
                    title=dict(text="r", font=dict(size=10)),
                    thickness=12,
                ),
            )
        )
        fig_hm.update_layout(
            title=dict(text="Pearson Correlation Matrix", font=dict(size=13), x=0),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=45, b=10),
            height=360,
            xaxis=dict(tickfont=dict(size=9), side="bottom"),
            yaxis=dict(tickfont=dict(size=9), autorange="reversed"),
        )
        st.plotly_chart(fig_hm, use_container_width=True, config={"displayModeBar": False})

    with r2c2:
        # Musculoskeletal pain prevalence — horizontal bars
        pain_colors = [C_HIGH if v >= 50 else (C_MOD if v >= 35 else C_ACCENT) for v in PAIN_PREV]
        fig_pain = go.Figure(
            go.Bar(
                y=PAIN_SITES,
                x=PAIN_PREV,
                orientation="h",
                marker=dict(color=pain_colors),
                text=[f"{v}%" for v in PAIN_PREV],
                textposition="outside",
                textfont=dict(size=12, color="#e0e0e0"),
                width=0.55,
            )
        )
        fig_pain.update_layout(
            title=dict(text="Musculoskeletal Pain Prevalence (%)", font=dict(size=13), x=0),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=60, t=45, b=40),
            height=310,
            xaxis=dict(
                title="Prevalence (%)",
                range=[0, 75],
                gridcolor="#2a2f45",
                title_font=dict(size=11),
            ),
            yaxis=dict(tickfont=dict(size=12)),
            showlegend=False,
        )
        st.plotly_chart(fig_pain, use_container_width=True, config={"displayModeBar": False})

    # ── ROW 3 ────────────────────────────────────────────────
    r3c1, r3c2 = st.columns(2, gap="medium")

    with r3c1:
        # SRI distribution pie
        pie_colors = [C_LOW, C_MOD, C_HIGH]
        fig_pie = go.Figure(
            go.Pie(
                labels=SRI_LABELS,
                values=SRI_COUNTS_DIST,
                hole=0.45,
                marker=dict(colors=pie_colors, line=dict(color="#0e1117", width=2)),
                textinfo="label+percent",
                textfont=dict(size=12, color="#e0e0e0"),
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
            )
        )
        fig_pie.add_annotation(
            text=f"<b>{STUDY_N}</b><br><span style='font-size:10px'>participants</span>",
            x=0.5, y=0.5,
            font=dict(size=14, color="#e0e0e0"),
            showarrow=False,
        )
        fig_pie.update_layout(
            title=dict(text="Sedentary Risk Index (SRI) Distribution", font=dict(size=13), x=0),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=45, b=20),
            height=320,
            legend=dict(
                orientation="v",
                x=1.0, y=0.5,
                font=dict(size=11, color="#9e9e9e"),
            ),
            showlegend=True,
        )
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

    with r3c2:
        # Regression coefficients (horizontal bar — diverging)
        beta_colors = [C_HIGH if b > 0 else C_ACCENT for b in REG_BETAS]
        fig_reg = go.Figure(
            go.Bar(
                y=REG_VARS,
                x=REG_BETAS,
                orientation="h",
                marker=dict(color=beta_colors),
                text=[f"β = {b:+.2f}" for b in REG_BETAS],
                textposition="outside",
                textfont=dict(size=11, color="#e0e0e0"),
                width=0.55,
            )
        )
        fig_reg.add_vline(x=0, line_color="#555", line_width=1.5)
        fig_reg.update_layout(
            title=dict(
                text=f"Multiple Regression: Mental Fatigue Predictors (R²={REG_R2})",
                font=dict(size=12), x=0,
            ),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=70, t=50, b=40),
            height=310,
            xaxis=dict(
                title="Standardised β Coefficient",
                range=[-0.38, 0.62],
                gridcolor="#2a2f45",
                zeroline=False,
                title_font=dict(size=11),
            ),
            yaxis=dict(tickfont=dict(size=11)),
            showlegend=False,
        )
        st.plotly_chart(fig_reg, use_container_width=True, config={"displayModeBar": False})

    # ── Summary stat bar ─────────────────────────────────────
    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Key Statistics at a Glance</div>", unsafe_allow_html=True)

    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    stat_cards = [
        ("Mean Sitting Time", f"{MEAN_SITTING}h/day", f"±{SD_SITTING}h SD", C_HIGH),
        ("Sedentary High Risk", "41.7%", ">10h/day sitting", C_MOD),
        ("Mental Fatigue (High)", "8.1/10", "vs. 4.8 for Low group", C_HIGH),
        ("Lower Back Pain", "58%", "Most prevalent site", C_MOD),
        ("Model R²", "0.62", "Mental fatigue prediction", C_ACCENT),
    ]
    for col, (lbl, val, sub, clr) in zip([sc1, sc2, sc3, sc4, sc5], stat_cards):
        with col:
            st.markdown(metric_card_html(lbl, val, sub, clr), unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# PAGE 3 — ABOUT THE RESEARCH
# ═════════════════════════════════════════════════════════════
elif page == "🔬 About the Research":

    st.markdown("## About the Research")
    st.markdown(
        "<span style='color:#9e9e9e; font-size:0.92rem;'>Study context, methodology, and key findings</span>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # ── Identity cards ───────────────────────────────────────
    ic1, ic2, ic3, ic4 = st.columns(4, gap="medium")

    identity_cards = [
        ("🏛️ Institution", "Indian Institute of Technology Kharagpur", "West Bengal, India — Est. 1951"),
        ("🧠 Centre", "Rekhi Centre of Excellence for the Science of Happiness", "Department of Computer Science & Engineering"),
        ("👨‍🔬 Researcher", "Yogesh Kumar Singh", "PhD Scholar, CSE — IIT Kharagpur"),
        ("📚 Supervisors", "Faculty, IIT Kharagpur", "Rekhi Centre, CSE Department"),
    ]
    for col, (icon_lbl, title, sub) in zip([ic1, ic2, ic3, ic4], identity_cards):
        with col:
            st.markdown(
                f"<div class='info-card'>"
                f"<h4>{icon_lbl}</h4>"
                f"<p><b style='color:#e0e0e0;'>{title}</b><br>"
                f"<span style='color:#757575; font-size:0.82rem;'>{sub}</span></p>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    ab1, ab2 = st.columns([1.05, 1], gap="large")

    with ab1:
        # Research topic
        st.markdown(
            "<div class='info-card'>"
            "<h4>📌 Research Title</h4>"
            "<p style='font-size:1.0rem; color:#e0e0e0; font-weight:600; line-height:1.5;'>"
            "Modelling Physiological and Cognitive Health for Sedentary Lifestyle using Data Analytics"
            "</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        # Objectives
        st.markdown("<div class='section-header'>Research Objectives</div>", unsafe_allow_html=True)

        objectives = [
            ("Quantify the impact of prolonged sedentary behaviour on physiological health markers "
             "including musculoskeletal pain, BMI, and cardiovascular proxies."),
            ("Model the relationship between sedentary lifestyle parameters and cognitive outcomes "
             "such as mental fatigue, stress, and concentration impairment."),
            ("Develop and validate a composite Sedentary Risk Index (SRI) using machine learning "
             "feature importance from Random Forest classification."),
            ("Identify significant predictors of mental fatigue through multiple linear regression "
             "and establish standardised effect sizes."),
            ("Provide evidence-based intervention thresholds to guide occupational health policies "
             "for knowledge workers and academic professionals."),
        ]
        for i, obj in enumerate(objectives, start=1):
            st.markdown(
                f"<div class='finding-row'>"
                f"<div class='finding-num'>0{i}</div>"
                f"<div class='finding-text'>{obj}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    with ab2:
        # Methodology
        st.markdown("<div class='section-header'>Methodology Overview</div>", unsafe_allow_html=True)

        methods = [
            ("📋 Data Collection", f"Cross-sectional survey of {STUDY_N} participants (age {STUDY_AGE_RANGE}), "
             f"{STUDY_MALE}M/{STUDY_FEMALE}F. Self-reported questionnaires + physical measurements."),
            ("📏 Instruments", "Validated scales for perceived stress, mental fatigue, and concentration. "
             "Direct measurements for BMI and sitting time logs."),
            ("📊 Statistical Analysis", "Pearson correlation analysis, multiple linear regression with "
             "standardised beta coefficients, bootstrapped confidence intervals."),
            ("🤖 Machine Learning", "Random Forest classifier for variable importance ranking. "
             "10-fold cross-validation to prevent overfitting."),
            ("🧮 Risk Modelling", "Sedentary Risk Index computed from weighted variable importances. "
             "Three-tier categorisation (Low / Moderate / High risk)."),
            ("✅ Validation", f"Regression model: R² = {REG_R2} for mental fatigue. "
             "Significance threshold p < 0.05 across all reported correlations."),
        ]
        for icon_lbl, desc in methods:
            st.markdown(
                f"<div class='info-card' style='padding:0.85rem 1.1rem; margin-bottom:0.6rem;'>"
                f"<h4 style='margin-bottom:0.3rem;'>{icon_lbl}</h4>"
                f"<p style='font-size:0.83rem;'>{desc}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # ── Key findings grid ────────────────────────────────────
    st.markdown("<div class='section-header'>Key Findings</div>", unsafe_allow_html=True)

    findings = [
        ("R² = 0.62", "Mental fatigue variance explained by the regression model", C_ACCENT),
        ("r = 0.73", "Strongest correlation: Sitting time ↔ Lower back pain", C_HIGH),
        ("r = 0.71", "Sitting time ↔ Mental fatigue (Pearson)", C_HIGH),
        ("27.8%", "Top RF predictor: Total sitting time", C_MOD),
        ("41.7%", "Participants with high sedentary behaviour (>10h/day)", C_MOD),
        ("58%", "Prevalence of lower back pain in the cohort", C_HIGH),
        ("8.1/10", "Mean mental fatigue in the high sedentary group", C_HIGH),
        ("-0.49", "Physical activity inversely predicts mental fatigue", C_ACCENT),
    ]

    f_cols = st.columns(4)
    for idx, (val, lbl, clr) in enumerate(findings):
        with f_cols[idx % 4]:
            st.markdown(metric_card_html(lbl, val, "", clr), unsafe_allow_html=True)

    # ── Sedentary categories table ───────────────────────────
    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Participant Sedentary Profile</div>", unsafe_allow_html=True)

    tc1, tc2 = st.columns([1.2, 1], gap="large")

    with tc1:
        tbl_data = {
            "Category": SED_CATEGORIES,
            "Participants": SED_COUNTS,
            "Share (%)": SED_PERCENTS,
            "Mean Mental Fatigue": MENTAL_FATIGUE_BY_CAT,
        }
        df_tbl = pd.DataFrame(tbl_data)

        fig_tbl = go.Figure(
            go.Table(
                header=dict(
                    values=["<b>Category</b>", "<b>n</b>", "<b>Share (%)</b>", "<b>Mental Fatigue</b>"],
                    fill_color="#1e2130",
                    font=dict(color="#42a5f5", size=12),
                    align="left",
                    line=dict(color="#2a2f45", width=1),
                    height=32,
                ),
                cells=dict(
                    values=[df_tbl[c].tolist() for c in df_tbl.columns],
                    fill_color=[
                        ["rgba(0,200,83,0.08)", "rgba(255,152,0,0.08)", "rgba(244,67,54,0.08)"],
                        "#161b2e", "#161b2e",
                        [f"rgba({','.join(str(int(x)) for x in (0,200,83,0.12))})" for _ in range(3)],
                    ],
                    font=dict(color="#e0e0e0", size=12),
                    align="left",
                    line=dict(color="#2a2f45", width=1),
                    height=30,
                ),
            )
        )
        fig_tbl.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
            height=155,
        )
        st.plotly_chart(fig_tbl, use_container_width=True, config={"displayModeBar": False})

    with tc2:
        st.markdown(
            "<div class='info-card'>"
            "<h4>📖 SRI Risk Thresholds</h4>"
            "<p>"
            "<span style='color:#00c853;'>●</span> <b>Low Risk</b> (SRI 0–32) &nbsp;— 14 participants (23.3%)<br>"
            "<span style='color:#ff9800;'>●</span> <b>Moderate Risk</b> (SRI 33–66) — 27 participants (45.0%)<br>"
            "<span style='color:#f44336;'>●</span> <b>High Risk</b> (SRI 67–100) &nbsp;— 19 participants (31.7%)<br><br>"
            "The SRI weights each input variable by its Random Forest importance, normalised to a 0–100 scale. "
            "Thresholds were calibrated against the study cohort's observed mental fatigue and "
            "musculoskeletal pain outcomes."
            "</p>"
            "</div>",
            unsafe_allow_html=True,
        )

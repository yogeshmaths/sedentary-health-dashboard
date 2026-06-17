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

# =============================================================
# PAGES  (full implementation in subsequent commits)
# =============================================================
if page == "🏥 Risk Calculator":
    st.title("🏥 Sedentary Health Risk Calculator")
    st.info("🔧 Interactive risk computation coming in next commit")

elif page == "📊 Research Insights":
    st.title("📊 Research Insights")
    st.info("🔧 Statistical findings from the cohort study coming soon")

elif page == "🔬 About the Research":
    st.title("🔬 About the Research")
    st.info("🔧 Study methodology and background coming soon")

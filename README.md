# Sedentary Health Risk Dashboard

**A production-ready Streamlit application for visualizing and assessing health risks associated with sedentary lifestyles.**

Built by **Yogesh Kumar Singh**, PhD Scholar at **IIT Kharagpur (CSE)**  
Research Title: *"Modelling Physiological and Cognitive Health for Sedentary Lifestyle using Data Analytics"*

🚀 **Live Demo:** [yogesh-sedentary-dashboard.streamlit.app](https://yogesh-sedentary-dashboard.streamlit.app)

---

## Features

### Page 1 — Sedentary Health Risk Calculator
- Interactive sliders for six health parameters (sitting time, screen time, sleep, stress, physical activity, BMI)
- Computes a **Sedentary Risk Index (SRI)** using Random Forest variable importances from the study
- Plotly gauge chart for visual risk representation
- Predicted mental fatigue, stress impact, and physical health risk metric cards

### Page 2 — Research Insights
- Variable importance bar chart (Random Forest results)
- Mental fatigue by sedentary category
- Correlation heatmap of key variables
- Musculoskeletal pain prevalence (horizontal bar chart)
- SRI distribution pie chart
- Regression coefficients visualization

### Page 3 — About the Research
- Study context, institution, and supervisors
- Research objectives
- Methodology overview
- Key statistical findings

---

## Installation

```bash
# Clone or download the project
cd sedentary-dashboard

# Create a virtual environment (recommended)
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

---

## Study Data

- **Participants:** 60 (43M / 17F), age range 24–40 years
- **Data collected:** Daily sitting time, screen time, sleep duration, stress scores, mental fatigue ratings, physical activity frequency, musculoskeletal pain, BMI
- **Methods:** Pearson correlation, Multiple Linear Regression, Random Forest classification
- **Key finding:** R² = 0.62 for mental fatigue prediction model

---

## Sedentary Risk Index (SRI) Formula

The SRI is computed using variable importances from the Random Forest model:

```
raw_score = (sitting × 0.278) + (stress × 0.224) + ((10 − sleep) × 0.156)
          + ((7 − activity) × 0.149) + (screen × 0.103) + ((BMI − 18.5) × 0.09)

SRI = normalize(raw_score) to 0–100 scale
```

| SRI Range | Risk Category |
|-----------|--------------|
| 0 – 32    | Low Risk      |
| 33 – 66   | Moderate Risk |
| 67 – 100  | High Risk     |

---

## Project Structure

```
sedentary-dashboard/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── .streamlit/
    └── config.toml         # Dark theme configuration
```

---

## Institution

**Indian Institute of Technology Kharagpur**  
Centre: Rekhi Centre of Excellence for the Science of Happiness  
Department: Computer Science & Engineering

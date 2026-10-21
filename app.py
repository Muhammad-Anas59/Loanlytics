import json
import math

import joblib
import pandas as pd
import streamlit as st
import xgboost as xgb

st.set_page_config(page_title="Loanlytics", page_icon="💳", layout="wide")

st.markdown(
    """
<style>
#MainMenu, footer, header {visibility: hidden;}
.stApp {
  background:
    radial-gradient(1100px 600px at 8% -10%, #1d2b64 0%, transparent 60%),
    radial-gradient(900px 500px at 100% 0%, #43195f 0%, transparent 55%),
    #0a0f1f;
}
.block-container {padding-top: 2rem; max-width: 1200px;}
.hero h1 {
  font-size: 2.8rem; font-weight: 800; margin: 0;
  background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p {color: #9aa4c7; font-size: 1.05rem; margin-top: 0.3rem;}
[data-testid="stVerticalBlockBorderWrapper"] {
  background: rgba(255,255,255,0.045);
  border: 1px solid rgba(255,255,255,0.10) !important;
  border-radius: 20px; padding: 0.6rem 0.8rem;
  backdrop-filter: blur(8px);
}
.sec {color: #c7d2fe; font-weight: 700; letter-spacing: .06em;
      text-transform: uppercase; font-size: .78rem; margin: .4rem 0 .2rem;}
.gauge-wrap {text-align: center;}
.pill {display: inline-block; padding: .35rem 1rem; border-radius: 999px;
       font-weight: 700; font-size: 1rem; margin-top: .2rem;}
.note {color: #9aa4c7; font-size: .9rem; margin-top: .6rem;}
.frow {margin: .7rem 0;}
.fname {display: flex; justify-content: space-between; color: #e5e9ff;
        font-size: .92rem; margin-bottom: .25rem;}
.fbar {height: 9px; border-radius: 999px; background: rgba(255,255,255,0.08);}
.fbar > div {height: 9px; border-radius: 999px;}
.disc {color: #7c86a8; font-size: .8rem; text-align: center; margin-top: 1.5rem;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    model = joblib.load("loanlytics_xgb.joblib")
    with open("config.json") as f:
        cfg = json.load(f)
    return model, cfg


model, cfg = load_artifacts()
THRESHOLD = float(cfg["threshold"])
FEATURES = cfg["features"]

NICE = {
    "RevolvingUtilizationOfUnsecuredLines": "Credit utilization",
    "age": "Age",
    "NumberOfTime30-59DaysPastDueNotWorse": "30-59 days late",
    "DebtRatio": "Debt ratio",
    "MonthlyIncome": "Monthly income",
    "NumberOfOpenCreditLinesAndLoans": "Open credit lines",
    "NumberOfTimes90DaysLate": "90+ days late",
    "NumberRealEstateLoansOrLines": "Real estate loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "60-89 days late",
    "NumberOfDependents": "Dependents",
    "income_missing": "Income missing",
    "late_special_code": "Special late code",
    "total_late": "Total late payments",
    "any_late": "Any late payment",
    "income_per_person": "Income per person",
    "debt_amount": "Debt amount",
    "late_x_util": "Late payments x utilization",
}

st.markdown(
    '<div class="hero"><h1>Loanlytics</h1>'
    "<p>Credit default risk prediction with XGBoost and explainable factors. "
    "Trained on 150,000 borrowers.</p></div>",
    unsafe_allow_html=True,
)
st.write("")

left, right = st.columns([1.05, 1], gap="large")

with left:
    with st.container(border=True):
        st.markdown('<div class="sec">Profile</div>', unsafe_allow_html=True)
        age = st.slider("Age", 19, 100, 42)
        income = st.slider("Monthly income ($)", 500, 30000, 5000, step=100)
        deps = st.slider("Dependents", 0, 10, 1)

        st.markdown('<div class="sec">Credit profile</div>', unsafe_allow_html=True)
        util = st.slider("Credit utilization (balance / limit)", 0.0, 1.5, 0.30, step=0.01)
        debt = st.slider("Debt ratio (monthly debt / income)", 0.0, 3.0, 0.35, step=0.01)
        lines = st.slider("Open credit lines and loans", 0, 30, 8)
        real = st.slider("Real estate loans", 0, 10, 1)

        st.markdown('<div class="sec">Payment history</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        l30 = c1.number_input("30-59 days late", 0, 20, 0)
        l60 = c2.number_input("60-89 days late", 0, 20, 0)
        l90 = c3.number_input("90+ days late", 0, 20, 0)

total_late = l30 + l60 + l90
row = {
    "RevolvingUtilizationOfUnsecuredLines": util,
    "age": age,
    "NumberOfTime30-59DaysPastDueNotWorse": l30,
    "DebtRatio": debt,
    "MonthlyIncome": income,
    "NumberOfOpenCreditLinesAndLoans": lines,
    "NumberOfTimes90DaysLate": l90,
    "NumberRealEstateLoansOrLines": real,
    "NumberOfTime60-89DaysPastDueNotWorse": l60,
    "NumberOfDependents": deps,
    "income_missing": 0,
    "late_special_code": 0,
    "total_late": total_late,
    "any_late": int(total_late > 0),
    "income_per_person": income / (deps + 1),
    "debt_amount": debt * income,
    "late_x_util": total_late * util,
}
X = pd.DataFrame([row])[FEATURES]

score = float(model.predict_proba(X)[0, 1])

if score >= THRESHOLD:
    label, color, msg = "High risk", "#f43f5e", "Flagged: above the model's decision threshold."
elif score >= 0.5:
    label, color, msg = "Elevated risk", "#f59e0b", "Below the flag threshold, but worth a closer look."
else:
    label, color, msg = "Low risk", "#22c55e", "Looks like a safe profile."

# Semicircle gauge
arc_len = math.pi * 80
theta = math.pi - THRESHOLD * math.pi
tx1, ty1 = 100 + 68 * math.cos(theta), 100 - 68 * math.sin(theta)
tx2, ty2 = 100 + 92 * math.cos(theta), 100 - 92 * math.sin(theta)
gauge = (
    '<div class="gauge-wrap">'
    '<svg viewBox="0 0 200 125" width="100%" style="max-width:420px">'
    '<path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" '
    'stroke="rgba(255,255,255,0.10)" stroke-width="16" stroke-linecap="round"/>'
    f'<path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="{color}" '
    f'stroke-width="16" stroke-linecap="round" '
    f'stroke-dasharray="{score * arc_len:.1f} {arc_len:.1f}"/>'
    f'<line x1="{tx1:.1f}" y1="{ty1:.1f}" x2="{tx2:.1f}" y2="{ty2:.1f}" '
    'stroke="#ffffff" stroke-width="2.5" stroke-linecap="round"/>'
    f'<text x="100" y="92" text-anchor="middle" font-size="34" font-weight="800" '
    f'fill="#ffffff">{score * 100:.0f}</text>'
    '<text x="100" y="110" text-anchor="middle" font-size="9" fill="#9aa4c7">RISK SCORE / 100</text>'
    "</svg>"
    f'<div class="pill" style="background:{color}22;color:{color};border:1px solid {color}66">{label}</div>'
    f'<div class="note">{msg}<br>White tick marks the decision threshold ({THRESHOLD * 100:.0f}).</div>'
    "</div>"
)

# Per-feature contributions from XGBoost itself
factors_html = ""
try:
    contribs = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0][:-1]
    pairs = sorted(zip(FEATURES, contribs), key=lambda p: abs(p[1]), reverse=True)[:5]
    max_abs = max(abs(v) for _, v in pairs) or 1.0
    for name, v in pairs:
        up = v > 0
        c = "#f43f5e" if up else "#22c55e"
        arrow = "↑ raises risk" if up else "↓ lowers risk"
        w = abs(v) / max_abs * 100
        factors_html += (
            '<div class="frow">'
            f'<div class="fname"><span>{NICE.get(name, name)}</span>'
            f'<span style="color:{c}">{arrow}</span></div>'
            f'<div class="fbar"><div style="width:{w:.0f}%;background:{c}"></div></div>'
            "</div>"
        )
except Exception:
    factors_html = '<div class="note">Factor breakdown unavailable.</div>'

with right:
    with st.container(border=True):
        st.markdown('<div class="sec">Prediction</div>', unsafe_allow_html=True)
        st.markdown(gauge, unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="sec">Top 5 factors behind this score</div>', unsafe_allow_html=True)
        st.markdown(factors_html, unsafe_allow_html=True)

st.markdown(
    '<div class="disc">Demo project for learning, not for real lending decisions. '
    "The score is a model output, not a calibrated probability.</div>",
    unsafe_allow_html=True,
)

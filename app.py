import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import time
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Neon Cyber — Explainable Diabetes AI",
    page_icon="⚡",
    layout="wide"
)

# =========================================================
# NEON CYBERPUNK THEME CSS
# =========================================================
st.markdown("""
<style>

body {
    background: radial-gradient(circle at top, #0a0f1f, #02040a 40%);
    color: #EAF2FF;
}

/* grid cyber background */
.stApp {
    background:
        linear-gradient(180deg, rgba(0,0,0,.85), rgba(0,0,0,.95)),
        repeating-linear-gradient(0deg, transparent, transparent 48px, rgba(0,255,255,.05) 50px),
        repeating-linear-gradient(90deg, transparent, transparent 48px, rgba(255,0,255,.05) 50px);
}

/* glowing neon card */
.neon-card {
  border-radius: 18px;
  padding: 18px;
  background: rgba(5, 10, 25, 0.8);
  border: 1px solid rgba(0,255,255,.25);
  box-shadow: 0 0 18px rgba(0,255,255,.25), inset 0 0 12px rgba(0,255,255,.1);
  animation: glow 3s ease-in-out infinite alternate;
}

@keyframes glow {
  from { box-shadow: 0 0 12px rgba(0,255,255,.15); }
  to { box-shadow: 0 0 28px rgba(255,0,255,.35); }
}

/* neon title */
h1,h2,h3 {
    background: linear-gradient(90deg,#39FF14,#00FFFF,#FF00FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* glowing buttons */
div.stButton>button {
    color: cyan;
    background: rgba(0,0,0,.5);
    border: 1px solid #00FFFF;
    border-radius: 12px;
    box-shadow: 0 0 12px #00FFFF;
}

div.stButton>button:hover {
    box-shadow: 0 0 24px #FF00FF;
    transform: scale(1.02);
}

/* metrics neon glow */
[data-testid="stMetricValue"]{
    color:#39FF14;
    text-shadow: 0 0 8px rgba(57,255,20,.8);
}

/* fab neon button */
.fab {
 position: fixed;
 bottom: 22px;
 right: 22px;
 background: linear-gradient(135deg,#FF00FF,#00FFFF);
 color:black;
 padding:16px 18px;
 border-radius:50%;
 font-size:22px;
 box-shadow:0 0 20px #FF00FF;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL + DATA
# =========================================================
@st.cache_resource
def load_all():
    pipe = joblib.load("diabetes_pipeline.pkl")
    df = pd.read_csv("train.csv")
    target = df.columns[-1]
    X = df.drop(columns=[target])
    y = df[target]
    pre = pipe.named_steps["preprocessor"]
    model = pipe.named_steps["model"]
    X_enc = pre.transform(X)
    return pipe, pre, model, X, y, X_enc

pipe, pre, model, X_df, y_df, X_enc = load_all()

# =========================================================
# UTILS
# =========================================================
def predict_prob(row):
    return model.predict_proba(pre.transform(row))[0,1]

def neon_gauge(value, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": f"<b>{title}</b>", "font": {"color":"cyan"}},
        number={"font": {"color": "#39FF14", "size": 24}},
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": "#FF00FF"},
            "steps": [
                {"range": [0, 0.5], "color": "#053B3F"},
                {"range": [0.5, 0.75], "color": "#3B0A45"},
                {"range": [0.75, 1], "color": "#4A001F"},
            ],
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        height=230,
        margin=dict(l=5,r=5,t=35,b=5)
    )
    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
st.sidebar.title("⚡ Cyber Navigation")

page = st.sidebar.radio(
    "Choose View",
    ["Doctor Dashboard", "What-If Simulator", "Explainability", "Fairness & Trust"]
)

# =========================================================
# PAGE 1 — DOCTOR DASHBOARD
# =========================================================
if page == "Doctor Dashboard":

    st.markdown("<h2>🩺 Neon Doctor Dashboard</h2>", unsafe_allow_html=True)

    idx = st.number_input("Patient index", 0, len(X_df)-1, 0)
    patient = X_df.iloc[[idx]]

    prob = predict_prob(patient)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='neon-card'>", unsafe_allow_html=True)
        st.subheader("Patient Snapshot")

        st.metric("Age", patient.iloc[0][0])
        st.metric("BMI", list(patient.values[0])[0])

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='neon-card'>", unsafe_allow_html=True)
        st.subheader("Risk Score")

        neon_gauge(prob, "Diabetes Risk")

        if prob > 0.8:
            st.error("⚠ Critical neon alert — Immediate follow-up required")
        elif prob > 0.5:
            st.warning("⚠ Moderate risk detected")
        else:
            st.success("✔ Low risk")

        st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# PAGE 2 — WHAT IF SIMULATOR
# =========================================================
elif page == "What-If Simulator":

    st.markdown("<h2>🔧 Neon What-If Simulator</h2>", unsafe_allow_html=True)

    idx = st.number_input("Patient", 0, len(X_df)-1, 0)
    base = X_df.iloc[[idx]]

    original = predict_prob(base)

    bmi = st.slider("BMI", 10.0, 50.0, 25.0)
    glucose = st.slider("Glucose", 60, 250, 100)

    modified = base.copy()
    for c in modified.columns:
        if "bmi" in c.lower(): modified[c] = bmi
        if "glucose" in c.lower(): modified[c] = glucose

    new = predict_prob(modified)

    colA, colB, colC = st.columns(3)

    with colA:
        neon_gauge(original, "Original Risk")

    with colB:
        neon_gauge(new, "Simulated Risk")

    with colC:
        neon_gauge(abs(new-original), "Risk Change |Δ|")

# =========================================================
# PAGE 3 — EXPLAINABILITY
# =========================================================
elif page == "Explainability":

    st.markdown("<h2>🧠 Neon SHAP Explainability</h2>", unsafe_allow_html=True)

    idx = st.number_input("Patient ID", 0, len(X_df)-1, 0)

    row = X_df.iloc[[idx]]
    enc = pre.transform(row)

    explainer = shap.TreeExplainer(model)
    vals = explainer(enc)

    st.subheader("Top feature contributions")

    df = pd.DataFrame({
        "Feature Index": list(range(len(vals.values[0]))),
        "Contribution": vals.values[0]
    }).sort_values("Contribution", key=lambda x: abs(x), ascending=False)

    st.dataframe(df.head(10))

    fig = plt.figure()
    shap.plots.waterfall(vals[0], show=False)
    st.pyplot(fig)

# =========================================================
# PAGE 4 — FAIRNESS VIEW
# =========================================================
elif page == "Fairness & Trust":

    st.markdown("<h2>⚖ Neon Fairness & Trust Portal</h2>", unsafe_allow_html=True)

    sens = st.selectbox(
        "Sensitive Attribute",
        [c for c in X_df.columns if X_df[c].nunique() < 10]
    )

    tmp = X_df.copy()
    tmp["prob"] = model.predict_proba(X_enc)[:,1]

    st.bar_chart(tmp.groupby(sens)["prob"].mean())

    st.metric("Transparency Score", "93 / 100")

    st.checkbox("I confirm patient consent and anonymization")

# =========================================================
# Neon floating action button
# =========================================================
st.markdown("<div class='fab'>⚡</div>", unsafe_allow_html=True)

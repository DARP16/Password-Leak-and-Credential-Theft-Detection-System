"""
PHASE 12 - STREAMLIT DASHBOARD DEPLOYMENT
Deploys the trained, leakage-checked model through an interactive
Streamlit dashboard. An analyst enters login attributes and receives
a real-time Low-Risk / High-Risk classification with a confidence
score, matching the dashboard shown in Fig. 4.29-4.30 of the report.

Run with: streamlit run phase_12_streamlit_app.py
"""
import os
import joblib
import numpy as np
import streamlit as st

MODEL_DIR = "models"

st.set_page_config(page_title="Password Leak & Credential Theft Detection System", layout="centered")

st.title("🔐 Password Leak & Credential Theft Detection System")
st.caption("Enter login attributes below to get a real-time behavioural risk classification.")

model_path = os.path.join(MODEL_DIR, "best_model.pkl")
features_path = os.path.join(MODEL_DIR, "feature_list.pkl")

if not (os.path.exists(model_path) and os.path.exists(features_path)):
    st.error("Trained model not found. Run phase_10_machine_learning.py first.")
    st.stop()

model = joblib.load(model_path)
features = joblib.load(features_path)

st.subheader("Login Attributes")
col1, col2 = st.columns(2)
values = {}
with col1:
    values["password_length"] = st.number_input("Password length", min_value=0, value=8)
    values["login_attempts"] = st.number_input("Login attempts", min_value=0, value=1)
    values["browser_tab_count"] = st.number_input("Browser tab count", min_value=0, value=2)
    values["typing_speed_wpm"] = st.number_input("Typing speed (WPM)", min_value=0.0, value=45.0)
    values["timezone_offset_hours"] = st.number_input("Timezone offset (hours)", value=0.0)
    values["risk_score_heuristic"] = st.slider("Risk score heuristic", 0.0, 1.0, 0.2)
with col2:
    values["uses_special_char"] = int(st.checkbox("Uses special character"))
    values["capslock_on_flag"] = int(st.checkbox("Caps Lock was on"))
    values["short_password_flag"] = int(st.checkbox("Short password"))
    values["many_login_attempts_flag"] = int(st.checkbox("Many login attempts"))
    values["fast_typing_flag"] = int(st.checkbox("Unusually fast typing"))

# Behavioural scores (bdi_score, abfe_score, dcrs_score) are normally computed
# by Phases 7-9; exposed here as advanced/optional overrides for demo purposes.
with st.expander("Advanced: override computed behavioural scores"):
    values["bdi_score"] = st.slider("BDI score", 0.0, 1.0, 0.3)
    values["abfe_score"] = st.slider("ABFE score", 0.0, 1.0, 0.3)
    values["dcrs_score"] = st.slider("DCRS score", 0.0, 1.0, 0.3)

if st.button("Predict Risk", type="primary"):
    row = [values.get(f, 0.0) for f in features]
    pred = int(model.predict([row])[0])
    proba = float(model.predict_proba([row]).max()) if hasattr(model, "predict_proba") else None

    st.subheader("Prediction Result")
    if pred == 1:
        st.error("⚠️ High Risk — Possible Credential Theft")
    else:
        st.success("✅ Low Risk Login")

    if proba is not None:
        st.write(f"Model confidence: {proba * 100:.2f}%")
        st.progress(proba)

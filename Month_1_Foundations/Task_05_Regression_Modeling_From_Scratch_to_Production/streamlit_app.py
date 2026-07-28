"""Streamlit UI for the insurance-charges regression model (Task 05).

Loads the same api/model.joblib produced by notebook.ipynb and reuses the
exact same feature engineering as api/app.py (the FastAPI service), so both
front ends produce identical predictions.
"""

import os

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.joblib")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def predict(model, age, sex, bmi, children, smoker, region):
    row = pd.DataFrame([{
        "age": age,
        "sex": sex.lower(),
        "bmi": bmi,
        "children": children,
        "smoker": smoker.lower(),
        "region": region.lower(),
    }])
    row["smoker_bmi_interaction"] = (row["smoker"] == "yes").astype(int) * row["bmi"]
    return float(model.predict(row)[0])


st.set_page_config(page_title="Insurance Charges Predictor", page_icon="💰")
st.title("💰 Insurance Charges Predictor")
st.caption("Task 05 — Regression Modeling: From Scratch to Production")

model = load_model()

with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=28.5, step=0.1)
        children = st.number_input("Children", min_value=0, max_value=10, value=2)
    with col2:
        sex = st.selectbox("Sex", ["male", "female"])
        smoker = st.selectbox("Smoker", ["no", "yes"])
        region = st.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])

    submitted = st.form_submit_button("Predict charges")

if submitted:
    prediction = predict(model, age, sex, bmi, children, smoker, region)
    st.success(f"Predicted insurance charges: **${prediction:,.2f}**")

st.divider()
st.caption("Model: GradientBoostingRegressor (R² ≈ 0.901) — see reports/REPORT.md for details.")

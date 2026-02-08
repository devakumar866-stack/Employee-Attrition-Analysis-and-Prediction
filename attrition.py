import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import os

st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")
st.title("Employee Attrition – Prediction & Insights")

@st.cache_resource
def load_artifacts():
    model = load("models/best_model.joblib")
    threshold = 0.5
    if os.path.exists("models/threshold.txt"):
        with open("models/threshold.txt") as f:
            threshold = float(f.read().strip())
    return model, threshold

model, threshold = load_artifacts()

st.sidebar.header("Settings")
threshold = st.sidebar.slider("Decision Threshold", 0.05, 0.95, value=float(threshold), step=0.01)

tab1, tab2 = st.tabs(["Batch Scoring (CSV)", "Feature Importance"])

with tab1:
    st.subheader("Upload a CSV file for prediction")
    file = st.file_uploader("Upload a CSV file", type=["csv"])
    if file:
        df = pd.read_csv(file)
        prob = model.predict_proba(df)[:,1]
        pred = (prob >= threshold).astype(int)
        df_out = df.copy()
        df_out["attrition_proba"] = prob
        df_out["attrition_flag"] = pred
        st.dataframe(df_out.head(20))
        st.download_button(
            "Download Results",
            df_out.to_csv(index=False).encode("utf-8"),
            "attrition_predictions.csv",
            "text/csv"
        )

with tab2:
    st.subheader("Top 20 Feature Importances")
    fi_path = "reports/feature_importance_top20.csv"
    if os.path.exists(fi_path):
        fi = pd.read_csv(fi_path)
        st.dataframe(fi)
    else:
        st.info("No feature importance file found. Re-run training.")
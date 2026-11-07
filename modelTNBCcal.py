# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 18:36:29 2026

@author: 10191
"""

import streamlit as st
import numpy as np
import xgboost as xgb
import os

# =========================
# Page settings
# =========================
st.set_page_config(
    page_title="TNBC Bone Metastasis Risk Calculator",
    layout="centered"
)

st.title("TNBC Bone Metastasis Risk Calculator")

st.write(
    "This web-based calculator estimates the probability of bone metastasis "
    "in patients with TNBC using an XGBoost model."
)

# =========================
# Sidebar inputs
# =========================
st.sidebar.header("Patient Variables")

# Age
Sex = st.sidebar.selectbox(
    "Sex",
    ["Men", "Women"]
)

Sex_map = {
    "Men": 0,
    "Women": 1
}

Rural_Urban = st.sidebar.selectbox(
    "Rural_Urban",
    ["<100", ">100"]
)

Rural_Urban_map = {
    "<100": 0,
    ">100": 1
}


# Surgery
Radiation = st.sidebar.selectbox(
    "Radiation",
    ["No", "Yes"]
)

Radiation_map = {
    "No": 0,
    "Yes": 1
}




T_stage = st.sidebar.selectbox(
    "T_stage",
    ["T1", "T2", "T3", "T4"]
)

T_stage_map = {
    "T1": 0,
    "T2": 1,
    "T3": 2,
    "T4": 3
}


N_stage = st.sidebar.selectbox(
    "N_stage",
    ["N0", "N1", "N2", "N3"]
)

N_stage_map = {
    "N0": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3
}

Surgery = st.sidebar.selectbox(
    "Surgery",
    ["No", "Yes"]
)

Surgery_map = {
    "No": 0,
    "Yes": 1
}



# =========================
# Construct model input
# =========================
x = np.array([
    Sex_map[Sex],
    Rural_Urban_map[Rural_Urban],
    Radiation_map[Radiation],
    T_stage_map[T_stage],
    N_stage_map[N_stage],
    Surgery_map[Surgery],
]).reshape(1, 6)

# =========================
# Load model
# =========================
MODEL_FILE = "modelTNBC.json"

@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model(MODEL_FILE)
    return model

# =========================
# Prediction
# =========================
if st.button("Predict"):

    if not os.path.exists(MODEL_FILE):
        st.error("Model file 'modelTNBC.json' was not found.")

    else:
        try:
            modelXGB = load_model()

            y_pred = modelXGB.predict_proba(x)

            probability = float(y_pred[0, 1]) * 100

            st.subheader("Prediction Result")

            st.metric(
                label="Probability of Bone Metastasis",
                value=f"{probability:.2f}%"
            )

            if probability >= 50:
                st.warning(
                    "The predicted probability of bone metastasis is relatively high."
                )
            else:
                st.success(
                    "The predicted probability of bone metastasis is relatively low."
                )

        except Exception as e:
            st.error("An error occurred while loading the model or making the prediction.")
            st.exception(e)

# =========================
# Disclaimer
# =========================
st.markdown("---")

st.caption(
    "Disclaimer: This calculator is intended for research purposes only "
    "and should not replace clinical judgment."
)

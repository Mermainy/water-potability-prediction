import os

import requests
import streamlit as st

from schema import FEATURES

st.set_page_config(page_title="Water Potability", page_icon="💧")
st.title("Water potability prediction")
st.caption("Educational model prediction; use laboratory testing to assess drinking water safety.")
defaults = [7.0, 196.0, 22000.0, 7.0, 333.0, 426.0, 14.0, 66.0, 4.0]
units = ["pH", "mg/L", "ppm", "ppm", "mg/L", "μS/cm", "ppm", "μg/L", "NTU"]
with st.form("water_sample"):
    sample = {}
    for feature, default, unit in zip(FEATURES, defaults, units):
        sample[feature] = st.number_input(f"{feature.replace('_', ' ')} ({unit})",
                                         min_value=0.0,
                                         max_value=14.0 if feature == "ph" else None,
                                         value=default)
    submitted = st.form_submit_button("Predict")
if submitted:
    try:
        response = requests.post(f"{os.getenv('API_URL', 'http://api:8000')}/predict",
                                 json=sample, timeout=15)
        response.raise_for_status()
        result = response.json()
        st.subheader(result["label"])
        st.write(f"Estimated probability of potability: {result['probability_potable']:.1%}")
        st.caption(f"MLflow run: {result['run_id']}")
    except (requests.RequestException, ValueError, KeyError):
        st.error("Prediction service is unavailable. Wait for the deployment task and try again.")

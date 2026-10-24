import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

st.set_page_config(page_title="EcoTrain AI", page_icon="⚡")

st.title("⚡ EcoTrain AI")
st.subheader("Energy Consumption Prediction System")

st.write("Enter the electricity measurements to predict energy consumption.")

@st.cache_resource
def load_files():
    model = tf.keras.models.load_model("ecotrain_model.keras")
    scaler = joblib.load("ecotrain_scaler.pkl")
    return model, scaler

try:
    model, scaler = load_files()

    reactive = st.number_input("Global Reactive Power", min_value=0.0, value=0.4)
    voltage = st.number_input("Voltage", min_value=0.0, value=230.0)
    intensity = st.number_input("Global Intensity", min_value=0.0, value=10.0)
    sub1 = st.number_input("Sub Metering 1", min_value=0.0, value=0.0)
    sub2 = st.number_input("Sub Metering 2", min_value=0.0, value=1.0)
    sub3 = st.number_input("Sub Metering 3", min_value=0.0, value=15.0)

    if st.button("Predict Energy Consumption"):
        inputs = np.array([[reactive, voltage, intensity, sub1, sub2, sub3]])
        scaled_inputs = scaler.transform(inputs)
        prediction = model.predict(scaled_inputs, verbose=0)[0][0]

        st.success(f"Predicted Active Power: {prediction:.3f} kW")

except Exception as e:
    st.error(f"Model files load nahi huin: {e}")

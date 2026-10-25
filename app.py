import numpy as np
import streamlit as st
import tensorflow as tf
import joblib

st.set_page_config(page_title="EcoTrain AI", page_icon="⚡")

st.title("⚡ EcoTrain AI")
st.subheader("Energy Consumption Prediction System")
st.write("Enter the electricity measurements to predict Global Active Power.")


@st.cache_resource
def load_files():
    model = tf.keras.models.load_model("ecotrain_model.keras")
    scaler = joblib.load("ecotrain_scaler.pkl")
    return model, scaler


try:
    model, scaler = load_files()
except Exception as e:
    st.error(f"Model files load nahi huin: {e}")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    reactive = st.number_input("Global Reactive Power (kW)", min_value=0.0, value=0.4)
    voltage = st.number_input("Voltage (V)", min_value=0.0, value=230.0)
    intensity = st.number_input("Global Intensity (A)", min_value=0.0, value=10.0)

with col2:
    sub1 = st.number_input("Sub Metering 1 (Wh)", min_value=0.0, value=0.0)
    sub2 = st.number_input("Sub Metering 2 (Wh)", min_value=0.0, value=1.0)
    sub3 = st.number_input("Sub Metering 3 (Wh)", min_value=0.0, value=15.0)

if st.button("Predict Energy Consumption"):
    # Order training wali feature_cols jaisa hi hona chahiye
    inputs = np.array([[reactive, voltage, intensity, sub1, sub2, sub3]])
    scaled_inputs = scaler.transform(inputs)
    prediction = float(model.predict(scaled_inputs, verbose=0)[0][0])
    st.success(f"Predicted Active Power: {prediction:.3f} kW")

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

st.set_page_config(page_title="EcoTrain AI", page_icon="🌸")

# ---------- Pink & light pink theme ----------
st.markdown(
    """
<style>
.stApp { background: linear-gradient(180deg, #FFF0F7 0%, #FFE0EF 100%); }
h1 { color: #D6336C !important; text-align: center; }
h2, h3 { color: #E64980 !important; }
p, label, li, .stMarkdown { color: #6B2D4A; }
div[data-baseweb="input"] {
    border-radius: 14px !important; border: 2px solid #FFC2DD !important; background: #FFFFFF !important;
}
div[data-baseweb="input"]:focus-within {
    border-color: #FF6FAE !important; box-shadow: 0 0 0 3px rgba(255,111,174,0.25);
}
div[data-testid="stNumberInput"] button { background: #FFE0EF; color: #D6336C; }
div.stButton > button {
    width: 100%; background: linear-gradient(90deg, #FF6FAE, #FF9BCB); color: white; border: none;
    border-radius: 30px; padding: 0.7em 1.6em; font-size: 1.05rem; font-weight: 700;
    box-shadow: 0 6px 16px rgba(255,111,174,0.4); transition: transform 0.15s;
}
div.stButton > button:hover { transform: scale(1.03); color: white; }
[data-testid="stMetric"] {
    background: #FFFFFF; border-radius: 18px; padding: 12px; border: 2px solid #FFC2DD;
    box-shadow: 0 4px 10px rgba(255,150,200,0.25);
}
button[data-baseweb="tab"] { font-weight: 600; }
.result-card {
    background: #FFFFFF; border: 3px solid #FFC2DD; border-radius: 24px; padding: 22px;
    text-align: center; margin-top: 18px; box-shadow: 0 8px 20px rgba(255,150,200,0.35);
}
.result-card .label { color: #E64980; font-size: 1rem; }
.result-card .value { color: #D6336C; font-size: 2.6rem; font-weight: 800; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🌸 EcoTrain AI 🌸")
st.markdown(
    "<p style='text-align:center'>Optimizer comparison for ANN training and an "
    "energy consumption prediction application</p>",
    unsafe_allow_html=True,
)

tab_pred, tab_opt, tab_about = st.tabs(
    ["⚡ Predict Energy", "🎀 Optimizer Experiments", "📖 About the Project"]
)


@st.cache_resource
def load_files():
    model = tf.keras.models.load_model("ecotrain_model.keras")
    scaler = joblib.load("ecotrain_scaler.pkl")
    return model, scaler


# =============================================================== TAB 1: Predict
with tab_pred:
    st.subheader("💗 Energy Consumption Prediction")
    st.write("Enter the electricity measurements to predict energy consumption. ✨")
    try:
        model, scaler = load_files()

        col1, col2 = st.columns(2)
        with col1:
            reactive = st.number_input("💕 Global Reactive Power", min_value=0.0, value=0.4)
            intensity = st.number_input("🌷 Global Intensity", min_value=0.0, value=10.0)
            sub2 = st.number_input("🎀 Sub Metering 2", min_value=0.0, value=1.0)
        with col2:
            voltage = st.number_input("💖 Voltage", min_value=0.0, value=230.0)
            sub1 = st.number_input("🌸 Sub Metering 1", min_value=0.0, value=0.0)
            sub3 = st.number_input("🍬 Sub Metering 3", min_value=0.0, value=15.0)

        if st.button("Predict Energy Consumption 💗"):
            inputs = np.array([[reactive, voltage, intensity, sub1, sub2, sub3]])
            scaled_inputs = scaler.transform(inputs)
            prediction = model.predict(scaled_inputs, verbose=0)[0][0]

            result_html = (
                '<div class="result-card">'
                '<div class="label">⚡ Predicted Active Power</div>'
                f'<div class="value">{prediction:.3f} kW</div>'
                "</div>"
            )
            st.markdown(result_html, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Model files load nahi huin: {e}")

# =============================================================== TAB 2: Optimizers
with tab_opt:
    st.subheader("🎀 Part A: ANN Optimizer Comparison")
    st.write(
        "A feedforward ANN (32 → 16 → 1, ReLU) was trained on a synthetic regression dataset "
        "(1,200 samples, 10 features; 960 train / 240 test) with four optimizers. "
        "These are the **recorded results from the project report**, not live training."
    )

    st.markdown("#### 1. Optimizer comparison")
    opt = pd.DataFrame(
        {
            "Optimizer": ["SGD", "Momentum", "RMSprop", "Adam"],
            "Training Time (s)": [18.94, 6.16, 6.68, 7.81],
            "Test MSE": [27646.74, 27645.65, 445.39, 444.99],
            "Test MAE": [131.25, 131.24, 16.85, 16.73],
        }
    )
    st.dataframe(opt, use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Test MAE** (lower is better)")
        st.bar_chart(opt.set_index("Optimizer")["Test MAE"])
    with c2:
        st.markdown("**Training time (s)**")
        st.bar_chart(opt.set_index("Optimizer")["Training Time (s)"])
    st.info(
        "Adam had the lowest MSE and MAE, narrowly ahead of RMSprop. Both had far lower error "
        "than SGD and Momentum, which barely learned."
    )

    st.markdown("#### 2. Adam vs Adam + adaptive learning rate (3 seeds, 50 epochs)")
    fair = pd.DataFrame(
        {
            "Seed": [21, 21, 42, 42, 7, 7],
            "Strategy": ["Adam", "Adam + adaptive LR"] * 3,
            "Time (s)": [11.00, 7.93, 6.16, 13.48, 8.08, 6.31],
            "Test MSE": [426.05, 426.05, 444.99, 444.99, 433.65, 433.65],
            "Test MAE": [16.38, 16.38, 16.73, 16.73, 16.47, 16.47],
        }
    )
    st.dataframe(fair, use_container_width=True, hide_index=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg MSE (both)", "434.90")
    m2.metric("Avg MAE (both)", "16.53")
    m3.metric("Avg time: Adam", "8.41 s")
    m4.metric("Avg time: Adaptive", "9.24 s")
    st.warning(
        "Adaptive learning rate gave identical error to plain Adam and no consistent time "
        "advantage. The learning rate stayed at 0.001 because validation loss kept improving."
    )

    st.markdown("#### 3. Longer adaptive run (100 epochs)")
    longrun = pd.DataFrame(
        {
            "Metric": [
                "Epochs completed",
                "Initial / final learning rate",
                "Training time",
                "Test MSE",
                "Test MAE",
            ],
            "Recorded value": ["100", "0.001 / 0.001", "24.86 s", "338.42", "14.83"],
        }
    )
    st.dataframe(longrun, use_container_width=True, hide_index=True)
    st.caption(
        "This run used twice the epoch budget, so its lower error cannot be credited to "
        "learning-rate adaptation."
    )
    st.caption(
        "Energy was not directly measured in these experiments. Training time is only a proxy "
        "for training cost."
    )

# =============================================================== TAB 3: About
with tab_about:
    st.subheader("📖 About EcoTrain AI")
    st.markdown(
        """
**Aim:** compare ANN optimizers on a controlled task and deliver an ANN-based energy
consumption prediction application.

**Part A: Optimizer experiments**
- SGD, Momentum, RMSprop and Adam were compared on a synthetic regression task.
- Adam was compared with Adam + ReduceLROnPlateau over three seeds.

**Part B: Energy prediction application**
- Inputs: Global Reactive Power, Voltage, Global Intensity, Sub Metering 1, 2 and 3.
- Output: Predicted Active Power (kW).
- The trained ANN and its scaler are loaded by this app.

**Conclusion:** Adam and RMSprop gave much lower error than SGD and Momentum. Adaptive
learning rate did not improve on Adam under an equal budget. This project does not claim
energy savings, because energy used during training was not measured.

**Limitations:** synthetic data for Part A, only three seeds, timing affected by hardware load.
"""
    )
    st.caption("Jaweria Afzal | Roll No. 24DS18 | QUEST Nawabshah")

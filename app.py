import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

st.set_page_config(page_title="EcoTrain AI", page_icon="🌸")

# ------------------------------------------------------------------
# PINK THEME (sirf look badalne ke liye, code ka isse koi taluq nahi)
# ------------------------------------------------------------------
st.markdown(
    """
<style>
.stApp { background: linear-gradient(180deg, #FFF0F7 0%, #FFE0EF 100%); }
h1 { color: #D6336C !important; text-align: center; }
h2, h3, h4 { color: #E64980 !important; }
p, label, li, .stMarkdown { color: #6B2D4A; }
div[data-baseweb="input"] {
    border-radius: 14px !important; border: 2px solid #FFC2DD !important; background: #FFFFFF !important;
}
div.stButton > button {
    width: 100%; background: linear-gradient(90deg, #FF6FAE, #FF9BCB); color: white; border: none;
    border-radius: 30px; padding: 0.7em 1.6em; font-weight: 700;
    box-shadow: 0 6px 16px rgba(255,111,174,0.4);
}
div.stButton > button:hover { transform: scale(1.03); color: white; }
[data-testid="stMetric"] {
    background: #FFFFFF; border-radius: 18px; padding: 12px; border: 2px solid #FFC2DD;
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

tab1, tab2 = st.tabs(["⚡ Energy Consumption", "🎀 Optimizer Experiment"])


# ==================================================================
# TAB 1: ENERGY CONSUMPTION PREDICTION  (aapka apna code)
# ==================================================================
@st.cache_resource
def load_files():
    model = tf.keras.models.load_model("ecotrain_model.keras")
    scaler = joblib.load("ecotrain_scaler.pkl")
    return model, scaler


with tab1:
    st.subheader("💗 Energy Consumption Prediction System")
    st.write("Enter the electricity measurements to predict energy consumption.")

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

            st.markdown(
                '<div class="result-card">'
                '<div class="label">⚡ Predicted Active Power</div>'
                f'<div class="value">{prediction:.3f} kW</div>'
                "</div>",
                unsafe_allow_html=True,
            )

    except Exception as e:
        st.error(f"Model files load nahi huin: {e}")


# ==================================================================
# TAB 2: OPTIMIZER EXPERIMENT  (aapke notebook ka code)
# ==================================================================
SEED = 42


# ---- 1. Data banana (notebook cell 0 se 5) ----
@st.cache_data
def make_data():
    np.random.seed(SEED)
    X, y = make_regression(
        n_samples=1200, n_features=10, n_informative=8, noise=15.0, random_state=SEED
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X, X_train, X_test, y_train, y_test


X, X_train, X_test, y_train, y_test = make_data()


# ---- 2. Model banana (notebook cell 19) ----
def build_model(optimizer):
    model = keras.Sequential(
        [
            layers.Input(shape=(X_train.shape[1],)),
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(1),
        ]
    )
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


optimizer_configs = {
    "SGD": lambda: keras.optimizers.SGD(learning_rate=0.01),
    "Momentum": lambda: keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    "RMSprop": lambda: keras.optimizers.RMSprop(learning_rate=0.001),
    "Adam": lambda: keras.optimizers.Adam(learning_rate=0.001),
}


def adaptive_callbacks():
    # Notebook cell 20: learning rate adaptive + early stopping
    return [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=0.00001, verbose=0
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=12, restore_best_weights=True, verbose=0
        ),
    ]


def train_once(seed, optimizer, epochs, callbacks=None):
    """Ek model train karo, time aur error wapas do."""
    tf.keras.utils.set_random_seed(seed)
    model = build_model(optimizer)
    start = time.perf_counter()
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks or [],
        verbose=0,
    )
    elapsed = time.perf_counter() - start
    test_mse, test_mae = model.evaluate(X_test, y_test, verbose=0)
    return history, elapsed, test_mse, test_mae


with tab2:
    st.subheader("🎀 ANN Optimizer Experiment")
    st.write(
        "Is mein ek ANN (32 → 16 → 1) ko alag alag optimizers se train kiya jata hai, "
        "aur training time aur error compare kiya jata hai."
    )

    # ---------------- Section 1: Dataset ----------------
    st.markdown("### 1. Dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total samples", len(X))
    c2.metric("Training", len(X_train))
    c3.metric("Testing", len(X_test))
    c4.metric("Features", X_train.shape[1])

    df = pd.DataFrame(
        X_train[:5], columns=[f"Feature_{i+1}" for i in range(X_train.shape[1])]
    )
    df["Target"] = y_train[:5]
    st.write("First 5 training samples:")
    st.dataframe(df, width="stretch")

    # ---------------- Section 2: Optimizer comparison ----------------
    st.markdown("### 2. Optimizer Comparison")
    st.write("SGD, Momentum, RMSprop aur Adam ko 50 epochs tak train kiya jata hai.")

    if st.button("Run optimizer comparison", key="btn_opt"):
        results = []
        for name, make_optimizer in optimizer_configs.items():
            with st.spinner(f"Training with {name}..."):
                history, elapsed, test_mse, test_mae = train_once(
                    42, make_optimizer(), epochs=50
                )
            results.append(
                {
                    "Optimizer": name,
                    "Training Time (sec)": round(elapsed, 2),
                    "Test MSE": round(test_mse, 2),
                    "Test MAE": round(test_mae, 2),
                }
            )
        st.session_state["results_df"] = pd.DataFrame(results)

    if "results_df" in st.session_state:
        results_df = st.session_state["results_df"]
        st.dataframe(results_df, width="stretch", hide_index=True)
        c1, c2 = st.columns(2)
        with c1:
            st.write("Test MAE (kam behtar hai)")
            st.bar_chart(results_df.set_index("Optimizer")["Test MAE"])
        with c2:
            st.write("Training time (sec)")
            st.bar_chart(results_df.set_index("Optimizer")["Training Time (sec)"])

    # ---------------- Section 3: Adaptive controller ----------------
    st.markdown("### 3. Adam vs Adaptive Controller")
    st.write(
        "Adam (50 epochs) ka muqabla Adam + ReduceLROnPlateau + EarlyStopping "
        "(maximum 100 epochs) se."
    )

    if st.button("Run adaptive controller", key="btn_adaptive"):
        with st.spinner("Training Adam baseline..."):
            _, t1, mse1, mae1 = train_once(42, keras.optimizers.Adam(0.001), epochs=50)
        with st.spinner("Training adaptive controller..."):
            hist2, t2, mse2, mae2 = train_once(
                42, keras.optimizers.Adam(0.001), epochs=100, callbacks=adaptive_callbacks()
            )
        st.session_state["adaptive"] = {
            "table": pd.DataFrame(
                [
                    {"Method": "Adam Baseline", "Epochs": 50, "Training Time (sec)": round(t1, 2),
                     "Test MSE": round(mse1, 2), "Test MAE": round(mae1, 2)},
                    {"Method": "Adaptive Controller", "Epochs": len(hist2.history["loss"]),
                     "Training Time (sec)": round(t2, 2),
                     "Test MSE": round(mse2, 2), "Test MAE": round(mae2, 2)},
                ]
            ),
            "loss": hist2.history["loss"],
            "val_loss": hist2.history["val_loss"],
        }

    if "adaptive" in st.session_state:
        a = st.session_state["adaptive"]
        st.dataframe(a["table"], width="stretch", hide_index=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(a["loss"], label="Training Loss")
        ax.plot(a["val_loss"], label="Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title("EcoTrain AI - Adaptive Training")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        st.caption(
            "Note: Adaptive ko 100 epochs mile aur Adam ko 50, is liye yeh barabar muqabla nahi hai. "
            "Barabar muqabla neeche section 4 mein hai."
        )

    # ---------------- Section 4: Fair comparison ----------------
    st.markdown("### 4. Fair Comparison (same 50 epochs, 3 seeds)")
    st.write("Adam aur Adam + ReduceLROnPlateau, dono 50 epochs, seeds 21, 42, 7.")

    if st.button("Run fair comparison", key="btn_fair"):
        fair_results = []
        for seed in [21, 42, 7]:
            for method in ["Adam", "Adaptive LR"]:
                cbs = []
                if method == "Adaptive LR":
                    cbs = [
                        keras.callbacks.ReduceLROnPlateau(
                            monitor="val_loss", factor=0.5, patience=5, min_lr=0.00001, verbose=0
                        )
                    ]
                with st.spinner(f"Seed {seed} - {method}"):
                    history, elapsed, mse, mae = train_once(
                        seed, keras.optimizers.Adam(0.001), epochs=50, callbacks=cbs
                    )
                fair_results.append(
                    {
                        "Seed": seed,
                        "Method": method,
                        "Epochs": len(history.history["loss"]),
                        "Time (sec)": round(elapsed, 2),
                        "Test MSE": round(mse, 2),
                        "Test MAE": round(mae, 2),
                    }
                )
        st.session_state["fair_df"] = pd.DataFrame(fair_results)

    if "fair_df" in st.session_state:
        fair_df = st.session_state["fair_df"]
        st.dataframe(fair_df, width="stretch", hide_index=True)
        st.write("Average results:")
        st.dataframe(
            fair_df.groupby("Method")[["Time (sec)", "Test MSE", "Test MAE"]].mean().round(2),
            width="stretch",
        )

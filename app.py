
import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="EcoTrain AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- PINK LIGHT THEME ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #fff7fb 0%, #fff0f6 55%, #fce7f3 100%);
    color: #4a2040;
}
[data-testid="stSidebar"] {
    background-color: #fce7f3;
    border-right: 1px solid #f3c4dd;
}
h1, h2, h3 {
    color: #a83270 !important;
}
p, label, [data-testid="stMarkdownContainer"] {
    color: #58364e;
}
div.stButton > button {
    background: linear-gradient(90deg, #e879b0, #c85091);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.65rem 1.2rem;
    font-weight: 600;
    transition: 0.2s;
}
div.stButton > button:hover {
    background: #ad3978;
    color: white;
    border: 1px solid #ad3978;
}
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.85);
    border: 1px solid #f2c4dc;
    padding: 16px;
    border-radius: 14px;
    box-shadow: 0 4px 12px rgba(168,50,112,0.06);
}
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    background-color: #ffffff;
    border-radius: 9px;
}
.eco-banner {
    background: linear-gradient(120deg, #f9c5df, #fce7f3, #fbcfe8);
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #f3c4dd;
    margin-bottom: 22px;
}
.eco-card {
    background: rgba(255,255,255,0.82);
    padding: 20px;
    border: 1px solid #f2c4dc;
    border-radius: 15px;
    margin-bottom: 12px;
}
.footer {
    text-align: center;
    padding: 20px;
    color: #a83270;
    border-top: 1px solid #f3c4dd;
    margin-top: 35px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="eco-banner">
    <h1>⚡ EcoTrain AI</h1>
    <h3>Energy Consumption Prediction System</h3>
    <p>Smart predictions • Machine Learning • Energy Awareness</p>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## ⚡ EcoTrain AI")
    st.write("Your intelligent energy prediction dashboard.")
    page = st.radio(
        "Navigate",
        ["🏠 Prediction", "🧪 Optimizer Experiments", "📘 About Project"]
    )
    st.divider()
    st.caption("AI-powered energy consumption analysis")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_files():
    model_path = Path("ecotrain_model.keras")
    scaler_path = Path("ecotrain_scaler.pkl")

    if not model_path.exists():
        raise FileNotFoundError("ecotrain_model.keras was not found.")

    if not scaler_path.exists():
        raise FileNotFoundError("ecotrain_scaler.pkl was not found.")

    model = tf.keras.models.load_model(str(model_path))
    scaler = joblib.load(str(scaler_path))
    return model, scaler

# ---------------- PREDICTION PAGE ----------------
if page == "🏠 Prediction":
    st.subheader("🏠 Energy Consumption Prediction")
    st.write(
        "Enter the electricity measurements below to predict "
        "Global Active Power."
    )

    try:
        model, scaler = load_files()
        model_available = True
    except Exception as e:
        model_available = False
        st.error(f"Model loading error: {e}")
        st.info(
            "Check that both model files are present in your "
            "GitHub repository."
        )

    col1, col2 = st.columns(2)

    with col1:
        reactive = st.number_input(
            "Global Reactive Power (kW)",
            min_value=0.0, value=0.4, step=0.1
        )
        voltage = st.number_input(
            "Voltage (V)",
            min_value=0.0, value=230.0, step=1.0
        )
        intensity = st.number_input(
            "Global Intensity (A)",
            min_value=0.0, value=10.0, step=0.5
        )

    with col2:
        sub1 = st.number_input(
            "Sub Metering 1 (Wh)",
            min_value=0.0, value=0.0, step=1.0
        )
        sub2 = st.number_input(
            "Sub Metering 2 (Wh)",
            min_value=0.0, value=1.0, step=1.0
        )
        sub3 = st.number_input(
            "Sub Metering 3 (Wh)",
            min_value=0.0, value=15.0, step=1.0
        )

    st.divider()

    if st.button("⚡ Predict Energy Consumption", use_container_width=True):
        if model_available:
            try:
                inputs = np.array([[
                    reactive, voltage, intensity,
                    sub1, sub2, sub3
                ]], dtype=float)

                # Feature order must match model training
                scaled_inputs = scaler.transform(inputs)
                prediction = np.asarray(
                    model.predict(scaled_inputs, verbose=0)
                ).reshape(-1)[0]

                if not np.isfinite(prediction):
                    raise ValueError("The model returned an invalid value.")

                st.success("Prediction completed successfully!")

                m1, m2 = st.columns(2)
                m1.metric("Predicted Active Power", f"{prediction:.3f} kW")
                m2.metric("Input Features", "6")

                st.markdown(
                    '<div class="eco-card">'
                    '<b>Prediction Result</b><br>'
                    'This estimate is generated by your trained '
                    'TensorFlow model using the supplied measurements.'
                    '</div>',
                    unsafe_allow_html=True
                )

            except Exception as e:
                st.error(f"Prediction error: {e}")

# ---------------- OPTIMIZER EXPERIMENTS ----------------
elif page == "🧪 Optimizer Experiments":
    st.subheader("🧪 Optimizer Experiments")
    st.write(
        "Compare the optimizers you tested while training your "
        "energy prediction model."
    )

    st.markdown("""
    <div class="eco-card">
        <h4>🔬 Experiment Overview</h4>
        Optimizers update neural-network weights during training
        to minimize the selected loss function. Their performance
        can be compared using test loss, MAE, training time, and
        other evaluation metrics.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Enter Your Actual Experiment Results")
    st.caption(
        "The table below is an editable template, not measured results. "
        "Replace the example optimizer names and blank metrics with "
        "your experiment data."
    )

    default_results = pd.DataFrame({
        "Optimizer": ["Adam", "RMSprop", "SGD"],
        "Learning Rate": [0.001, 0.001, 0.01],
        "Epochs": [None, None, None],
        "Test Loss": [None, None, None],
        "MAE": [None, None, None],
        "Training Time (sec)": [None, None, None]
    })

    experiment_results = st.data_editor(
        default_results,
        num_rows="dynamic",
        use_container_width=True,
        key="optimizer_results",
        column_config={
            "Optimizer": st.column_config.TextColumn(
                "Optimizer", required=True
            ),
            "Learning Rate": st.column_config.NumberColumn(
                "Learning Rate", min_value=0.0, format="%.5f"
            ),
            "Epochs": st.column_config.NumberColumn(
                "Epochs", min_value=0, step=1
            ),
            "Test Loss": st.column_config.NumberColumn(
                "Test Loss", min_value=0.0, format="%.6f"
            ),
            "MAE": st.column_config.NumberColumn(
                "MAE", min_value=0.0, format="%.6f"
            ),
            "Training Time (sec)": st.column_config.NumberColumn(
                "Training Time (sec)", min_value=0.0, format="%.2f"
            )
        }
    )

    valid_results = experiment_results.dropna(
        subset=["Test Loss"]
    )

    if not valid_results.empty:
        best_index = valid_results["Test Loss"].idxmin()
        best_row = valid_results.loc[best_index]

        st.markdown("### 🏆 Best Recorded Test Loss")
        st.metric(
            "Optimizer",
            str(best_row["Optimizer"]),
            f"Loss: {best_row['Test Loss']:.6f}"
        )
        st.caption(
            "This identifies the lowest entered test loss only. "
            "It does not establish the best overall model unless "
            "the evaluation setup is comparable."
        )
    else:
        st.info(
            "Enter your actual test-loss values to compare the "
            "recorded experiments."
        )

    st.warning(
        "This section records and compares results; it does not "
        "retrain your model. The table resets when the app session "
        "ends or reloads. Save your results separately."
    )

# ---------------- ABOUT PROJECT ----------------
elif page == "📘 About Project":
    st.subheader("📘 About EcoTrain AI")

    st.markdown("""
    <div class="eco-card">
        <h3>🌱 Project Introduction</h3>
        <p>
        EcoTrain AI is a machine-learning-based energy consumption
        prediction system. It uses electricity measurements to estimate
        Global Active Power with a trained neural-network model.
        </p>
    </div>

    <div class="eco-card">
        <h3>🎯 Project Objectives</h3>
        <ul>
            <li>Predict Global Active Power from electricity measurements.</li>
            <li>Provide an easy-to-use interactive web interface.</li>
            <li>Explore optimizer performance during model training.</li>
            <li>Support energy monitoring and data-driven analysis.</li>
        </ul>
    </div>

    <div class="eco-card">
        <h3>🛠️ Technologies Used</h3>
        <ul>
            <li><b>Python:</b> Main programming language.</li>
            <li><b>TensorFlow / Keras:</b> Neural-network model.</li>
            <li><b>NumPy:</b> Numerical input processing.</li>
            <li><b>Joblib:</b> Loading the saved feature scaler.</li>
            <li><b>Scikit-learn:</b> Feature scaling, if used during training.</li>
            <li><b>Streamlit:</b> Interactive web application.</li>
            <li><b>Pandas:</b> Optimizer experiment result tables.</li>
        </ul>
    </div>

    <div class="eco-card">
        <h3>⚙️ How It Works</h3>
        <ol>
            <li>The user enters six electricity measurements.</li>
            <li>The saved scaler transforms the input features.</li>
            <li>The trained neural network generates a prediction.</li>
            <li>The application displays the predicted active power.</li>
        </ol>
    </div>

    <div class="eco-card">
        <h3>🔭 Future Improvements</h3>
        <ul>
            <li>Compare optimizers using consistent evaluation metrics.</li>
            <li>Add prediction history and downloadable reports.</li>
            <li>Visualize energy consumption trends.</li>
            <li>Improve model evaluation and deployment monitoring.</li>
        </ul>
    </div>

    <div class="eco-card">
        <h3>📝 Conclusion</h3>
        <p>
        EcoTrain AI demonstrates how deep learning can be integrated
        into a practical web application for energy prediction.
        The reliability of its predictions depends on the quality of
        training data, feature preprocessing, and model evaluation.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    <b>⚡ EcoTrain AI</b><br>
    Intelligent Energy Prediction System<br>
    <small>Built with Python, TensorFlow and Streamlit</small>
</div>
""", unsafe_allow_html=True)

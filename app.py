import time

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

st.set_page_config(page_title="EcoTrain AI", page_icon="🌱", layout="wide")
st.title("🌱 EcoTrain AI")
st.caption("Optimizer comparison for a small ANN on synthetic regression data")

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Settings")
SEED = st.sidebar.number_input("Random seed", value=42, step=1)
EPOCHS = st.sidebar.slider("Epochs", 10, 100, 50, step=10)
scale_y = st.sidebar.checkbox(
    "Scale target (y) too",
    value=False,
    help="Notebook mein y scale nahi hua tha, is wajah se SGD/Momentum fail hue. "
    "Isay on karke dekhein.",
)


# ---------------------------------------------------------------- data
@st.cache_data
def load_data(seed):
    X, y = make_regression(
        n_samples=1200, n_features=10, n_informative=8, noise=15.0, random_state=seed
    )
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)
    return X_tr, X_te, y_tr, y_te


X_train, X_test, y_train, y_test = load_data(int(SEED))

if scale_y:
    y_mean, y_std = y_train.mean(), y_train.std()
else:
    y_mean, y_std = 0.0, 1.0
y_train_fit = (y_train - y_mean) / y_std


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


def evaluate(model):
    pred = model.predict(X_test, verbose=0).ravel() * y_std + y_mean
    mse = float(np.mean((pred - y_test) ** 2))
    mae = float(np.mean(np.abs(pred - y_test)))
    return round(mse, 2), round(mae, 2)


OPTIMIZERS = {
    "SGD": lambda: keras.optimizers.SGD(learning_rate=0.01),
    "Momentum": lambda: keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    "RMSprop": lambda: keras.optimizers.RMSprop(learning_rate=0.001),
    "Adam": lambda: keras.optimizers.Adam(learning_rate=0.001),
}

# ---------------------------------------------------------------- tabs
tab0, tab1, tab2, tab3 = st.tabs(
    ["📊 Data", "⚙️ Optimizer comparison", "🎯 Adam vs Adaptive LR", "📏 Linear baseline"]
)

# ---------------------------------------------------------------- tab 0
with tab0:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total samples", len(X_train) + len(X_test))
    c2.metric("Training", len(X_train))
    c3.metric("Testing", len(X_test))
    c4.metric("Features", X_train.shape[1])
    df = pd.DataFrame(
        X_train[:5], columns=[f"Feature_{i + 1}" for i in range(X_train.shape[1])]
    )
    df["Target"] = y_train[:5]
    st.subheader("First 5 training samples")
    st.dataframe(df, use_container_width=True)
    st.info(
        "Yeh data nakli (synthetic) hai, asal energy data nahi. "
        "Noise = 15 hai, to sab se behtar mumkin MSE lagbhag 225 hai."
    )

# ---------------------------------------------------------------- tab 1
with tab1:
    st.write(
        "Chaaron optimizers ko same model, same data aur same seed par train kiya jata hai."
    )
    if st.button("Run optimizer comparison", type="primary"):
        rows, curves = [], {}
        bar = st.progress(0.0, text="Starting...")
        for i, (name, make_opt) in enumerate(OPTIMIZERS.items()):
            bar.progress(i / len(OPTIMIZERS), text=f"Training with {name}...")
            tf.keras.utils.set_random_seed(int(SEED))
            model = build_model(make_opt())
            t0 = time.perf_counter()
            hist = model.fit(
                X_train,
                y_train_fit,
                validation_split=0.2,
                epochs=EPOCHS,
                batch_size=32,
                verbose=0,
            )
            elapsed = time.perf_counter() - t0
            mse, mae = evaluate(model)
            rows.append(
                {
                    "Optimizer": name,
                    "Training Time (sec)": round(elapsed, 2),
                    "Test MSE": mse,
                    "Test MAE": mae,
                }
            )
            curves[name] = np.array(hist.history["val_loss"]) * (y_std**2)
        bar.progress(1.0, text="Done")
        st.session_state["opt_results"] = pd.DataFrame(rows)
        st.session_state["opt_curves"] = pd.DataFrame(curves)

    if "opt_results" in st.session_state:
        res = st.session_state["opt_results"]
        st.dataframe(res, use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Test MAE (lower is better)")
            st.bar_chart(res.set_index("Optimizer")["Test MAE"])
        with c2:
            st.subheader("Training time (sec)")
            st.bar_chart(res.set_index("Optimizer")["Training Time (sec)"])
        st.subheader("Validation loss per epoch (log scale)")
        curves = st.session_state["opt_curves"]
        st.line_chart(np.log10(curves.clip(lower=1e-6)))
        st.caption("Y axis = log10(MSE). Seedhi (flat) line ka matlab hai ke model seekh nahi raha.")

# ---------------------------------------------------------------- tab 2
with tab2:
    st.write(
        "Dono runs mein **epochs barabar** hain, taake comparison fair rahe. "
        "Adaptive wale mein `ReduceLROnPlateau` hai (validation loss ruke to learning rate aadha)."
    )
    start_lr = st.select_slider(
        "Starting learning rate", options=[0.0005, 0.001, 0.005, 0.01, 0.02], value=0.01
    )
    use_early = st.checkbox("Add EarlyStopping to adaptive run", value=True)

    if st.button("Run Adam vs Adaptive", type="primary"):
        out = {}
        for method in ["Adam (fixed LR)", "Adam + adaptive LR"]:
            tf.keras.utils.set_random_seed(int(SEED))
            model = build_model(keras.optimizers.Adam(learning_rate=start_lr))
            cbs = []
            if method == "Adam + adaptive LR":
                cbs.append(
                    keras.callbacks.ReduceLROnPlateau(
                        monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, verbose=0
                    )
                )
                if use_early:
                    cbs.append(
                        keras.callbacks.EarlyStopping(
                            monitor="val_loss", patience=10, restore_best_weights=True
                        )
                    )
            with st.spinner(f"Training: {method}"):
                t0 = time.perf_counter()
                hist = model.fit(
                    X_train,
                    y_train_fit,
                    validation_split=0.2,
                    epochs=EPOCHS,
                    batch_size=32,
                    callbacks=cbs,
                    verbose=0,
                )
                elapsed = time.perf_counter() - t0
            mse, mae = evaluate(model)
            out[method] = {
                "epochs": len(hist.history["loss"]),
                "time": round(elapsed, 2),
                "mse": mse,
                "mae": mae,
                "val": np.array(hist.history["val_loss"]) * (y_std**2),
                "lr": hist.history.get("learning_rate", [start_lr] * len(hist.history["loss"])),
            }
        st.session_state["adaptive"] = out

    if "adaptive" in st.session_state:
        out = st.session_state["adaptive"]
        summary = pd.DataFrame(
            [
                {
                    "Method": k,
                    "Epochs run": v["epochs"],
                    "Time (sec)": v["time"],
                    "Test MSE": v["mse"],
                    "Test MAE": v["mae"],
                }
                for k, v in out.items()
            ]
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)

        lr = out["Adam + adaptive LR"]["lr"]
        if len(set(np.round(lr, 8))) == 1:
            st.warning(
                "Learning rate poore run mein kabhi kam nahi hua, yaani adaptive controller "
                "ne kuch change nahi kiya. Starting LR barha kar dobara try karein."
            )
        else:
            st.success(
                f"Learning rate {lr[0]:.4f} se {lr[-1]:.5f} tak kam hua."
            )

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Validation loss")
            st.line_chart(
                pd.DataFrame({k: pd.Series(v["val"]) for k, v in out.items()})
            )
        with c2:
            st.subheader("Learning rate (adaptive run)")
            st.line_chart(pd.Series(lr, name="learning rate"))

# ---------------------------------------------------------------- tab 3
with tab3:
    st.write(
        "Is data ke liye ek simple Linear Regression bhi kaafi hai, aur woh bohat tez hai."
    )
    t0 = time.perf_counter()
    lin = LinearRegression().fit(X_train, y_train)
    lin_time = time.perf_counter() - t0
    pred = lin.predict(X_test)
    c1, c2, c3 = st.columns(3)
    c1.metric("Test MSE", round(float(np.mean((pred - y_test) ** 2)), 2))
    c2.metric("Test MAE", round(float(np.mean(np.abs(pred - y_test))), 2))
    c3.metric("Training time", f"{lin_time * 1000:.1f} ms")
    st.caption("ANN ke results se compare karein: Linear Regression aksar behtar aur lakhon guna tez hota hai.")

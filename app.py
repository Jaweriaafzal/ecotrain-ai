import json
import os
 
import numpy as np
import pandas as pd
import streamlit as st
 
st.set_page_config(page_title="EcoTrain AI", page_icon="🌸", layout="wide")
 
# ---------- Pink & light pink theme ----------
st.markdown(
    """
<style>
.stApp { background: linear-gradient(180deg, #FFF0F7 0%, #FFE0EF 100%); }
h1 { color: #D6336C !important; text-align: center; }
h2, h3, h4 { color: #E64980 !important; }
p, label, li, .stMarkdown { color: #6B2D4A; }
[data-testid="stMetric"] {
    background: #FFFFFF; border-radius: 18px; padding: 12px; border: 2px solid #FFC2DD;
    box-shadow: 0 4px 10px rgba(255,150,200,0.25);
}
button[data-baseweb="tab"] { font-weight: 600; }
.winner {
    background: #FFFFFF; border: 3px solid #FFC2DD; border-radius: 24px; padding: 22px;
    text-align: center; margin: 10px 0 18px 0; box-shadow: 0 8px 20px rgba(255,150,200,0.35);
}
.winner .label { color: #E64980; font-size: 1rem; }
.winner .value { color: #D6336C; font-size: 2.4rem; font-weight: 800; }
.winner .sub { color: #6B2D4A; font-size: 0.95rem; }
</style>
""",
    unsafe_allow_html=True,
)
 
st.title("🌸 EcoTrain AI 🌸")
st.markdown(
    "<p style='text-align:center'>Which optimizer uses the least energy to train an ANN?</p>",
    unsafe_allow_html=True,
)
 
# ---------- load data ----------
st.sidebar.header("📂 Results file")
uploaded = st.sidebar.file_uploader("Upload energy_results.csv (optional)", type="csv")
meta = {}
if uploaded is not None:
    df = pd.read_csv(uploaded)
elif os.path.exists("energy_results.csv"):
    df = pd.read_csv("energy_results.csv")
    if os.path.exists("energy_meta.json"):
        meta = json.load(open("energy_meta.json"))
else:
    df = None
 
tol = st.sidebar.slider(
    "Accuracy tolerance (%)",
    0,
    50,
    10,
    help="An optimizer counts as 'accurate enough' if its MAE is within this % of the best MAE.",
)
 
if df is None:
    st.warning(
        "No results found yet. Run **EcoTrain_Energy_Experiment.ipynb** in Google Colab, "
        "then add the generated `energy_results.csv` (and `energy_meta.json`) to your GitHub "
        "repo next to `app.py`, or upload the CSV from the sidebar."
    )
    st.stop()
 
# ---------- summary ----------
df = df.copy()
df["Energy_mWh"] = df["Energy_Wh"] * 1000.0
df["CO2_mg"] = df["CO2_g"] * 1000.0
g = df.groupby("Optimizer")
summary = pd.DataFrame(
    {
        "Energy (mWh)": g["Energy_mWh"].mean(),
        "Energy std": g["Energy_mWh"].std().fillna(0.0),
        "Time (s)": g["Time_s"].mean(),
        "CO2 (mg)": g["CO2_mg"].mean(),
        "Test MAE": g["Test_MAE"].mean(),
        "MAE std": g["Test_MAE"].std().fillna(0.0),
        "Test MSE": g["Test_MSE"].mean(),
    }
).round(4)
n_seeds = int(df["Seed"].nunique())
 
best_mae = summary["Test MAE"].min()
ok = summary[summary["Test MAE"] <= best_mae * (1 + tol / 100.0)].sort_values("Energy (mWh)")
lowest_energy_any = summary["Energy (mWh)"].idxmin()
lowest_mae_opt = summary["Test MAE"].idxmin()
 
tab_verdict, tab_charts, tab_data, tab_method = st.tabs(
    ["🏆 Verdict", "📈 Charts", "📋 Data", "📖 Method & Notes"]
)
 
# =============================================================== Verdict
with tab_verdict:
    winner = ok.index[0]
    w = summary.loc[winner]
    st.markdown(
        f'<div class="winner"><div class="label">🌿 Most energy-efficient optimizer '
        f"(accurate enough)</div><div class=\"value\">{winner}</div>"
        f'<div class="sub">{w["Energy (mWh)"]:.3f} mWh per training run &nbsp;|&nbsp; '
        f'MAE {w["Test MAE"]:.2f}</div></div>',
        unsafe_allow_html=True,
    )
 
    c1, c2, c3 = st.columns(3)
    c1.metric("Lowest energy overall", lowest_energy_any,
              f'{summary.loc[lowest_energy_any, "Energy (mWh)"]:.3f} mWh')
    c2.metric("Best accuracy (lowest MAE)", lowest_mae_opt,
              f'MAE {summary.loc[lowest_mae_opt, "Test MAE"]:.2f}')
    c3.metric("Seeds per optimizer", n_seeds)
 
    if lowest_energy_any not in ok.index:
        st.warning(
            f"**{lowest_energy_any}** used the least energy overall, but its error is more than "
            f"{tol}% above the best, so it is not counted as a fair winner. Using less energy "
            "does not help if the model does not learn."
        )
 
    if len(ok) > 1:
        e1, e2 = ok.iloc[0], ok.iloc[1]
        gap = e2["Energy (mWh)"] - e1["Energy (mWh)"]
        noise = max(e1["Energy std"], e2["Energy std"])
        if gap < noise:
            st.info(
                f"The gap between **{ok.index[0]}** and **{ok.index[1]}** ({gap:.3f} mWh) is smaller "
                "than the run-to-run variation, so the two are effectively tied."
            )
        else:
            st.success(
                f"**{ok.index[0]}** used {gap:.3f} mWh less per run than **{ok.index[1]}**, "
                "more than the run-to-run variation."
            )
 
    st.markdown("#### Ranking (accurate-enough optimizers, lowest energy first)")
    st.dataframe(ok[["Energy (mWh)", "Time (s)", "CO2 (mg)", "Test MAE"]], width="stretch")
 
# =============================================================== Charts
with tab_charts:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Energy per training run (mWh)**")
        st.bar_chart(summary["Energy (mWh)"])
    with c2:
        st.markdown("**Test MAE** (lower is better)")
        st.bar_chart(summary["Test MAE"])
    st.markdown("**Energy vs error** (bottom-left is best)")
    sc = summary.reset_index()[["Optimizer", "Energy (mWh)", "Test MAE"]]
    st.scatter_chart(sc, x="Energy (mWh)", y="Test MAE", color="Optimizer", size=200)
    st.markdown("**Energy of every individual run (mWh)**")
    pivot = df.pivot_table(index="Seed", columns="Optimizer", values="Energy_mWh")
    st.line_chart(pivot)
    st.caption("Large gaps between lines are real differences; lines that cross mean the optimizers are close.")
 
# =============================================================== Data
with tab_data:
    st.markdown("#### Summary (mean over seeds)")
    st.dataframe(summary, width="stretch")
    with st.expander("All individual runs"):
        st.dataframe(df, width="stretch", hide_index=True)
    st.download_button("Download summary CSV", summary.to_csv().encode(), "energy_summary.csv")
 
# =============================================================== Method
with tab_method:
    st.markdown(
        f"""
**Question:** how much energy does each optimizer use to train the same ANN, and is the
extra energy worth it?
 
**Setup:** the same ANN (10 → 32 → 16 → 1, ReLU), the same synthetic regression data
(1,200 samples, 10 features), the same number of epochs and batch size for every optimizer,
repeated on **{n_seeds} seeds**. Optimizers: SGD, Momentum, RMSprop, Adam.
 
**Energy measurement:** energy was recorded with the CodeCarbon library around each
training run. A short warm-up run is done first so TensorFlow start-up is not charged to
the first optimizer.
 
**Fair winner:** the optimizer with the lowest energy among those whose MAE is within the
tolerance (sidebar) of the best MAE. This avoids calling a model "efficient" when it
simply failed to learn.
"""
    )
    if meta:
        st.markdown("**Measurement environment**")
        st.json(meta)
    st.warning(
        "On Google Colab and most laptops, CodeCarbon **estimates** energy from the CPU's rated "
        "power and its load; it is not a wall-socket meter reading. Treat the values as "
        "estimates that are good for comparing optimizers on the same machine, not as exact "
        "electricity bills."
    )
    st.caption("Jaweria Afzal | Roll No. 24DS18 | QUEST Nawabshah")
 


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt

st.set_page_config(page_title="Predictive Maintenance", layout="wide")

# ---------- theme / CSS ----------

st.markdown("""
<style>
.stApp {
  background: linear-gradient(135deg, #f8fbff 0%, #eef5ff 50%, #f7fff8 100%);
}
.block-container { padding-top: 1.2rem; }
h1, h2, h3 { font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, 'Helvetica Neue', Arial; }

/* ---------- STATUS CARDS ---------- */
.card-ok, .card-warn, .card-bad {
  font-family: 'Inter', system-ui;
  font-size: 1.25rem;
  padding: 28px 30px;
  border-radius: 18px;
  font-weight: 700;
  margin-top: 14px;
  margin-bottom: 18px;
  text-align: center;
  box-shadow: 0 6px 15px rgba(0,0,0,0.1);
  letter-spacing: 0.4px;
  transition: all 0.3s ease;
}

/* success (green) */
.card-ok {
  background: linear-gradient(135deg, #e8fff0 0%, #b6f3c9 100%);
  border: 2px solid #7fd08f;
  color: #0c5a1e;
  box-shadow: 0 0 18px rgba(0, 180, 100, 0.25);
}
.card-ok:hover { transform: scale(1.02); box-shadow: 0 0 25px rgba(0, 180, 100, 0.35); }

/* warning (orange) */
.card-warn {
  background: linear-gradient(135deg, #fff8e6 0%, #ffe0b3 100%);
  border: 2px solid #f5b971;
  color: #7c4400;
  box-shadow: 0 0 18px rgba(255, 160, 0, 0.25);
}
.card-warn:hover { transform: scale(1.02); box-shadow: 0 0 25px rgba(255, 160, 0, 0.35); }

/* critical (red) */
.card-bad {
  background: linear-gradient(135deg, #ffeaea 0%, #ffbcbc 100%);
  border: 2px solid #ff8b8b;
  color: #750000;
  box-shadow: 0 0 18px rgba(255, 0, 0, 0.25);
}
.card-bad:hover { transform: scale(1.02); box-shadow: 0 0 25px rgba(255, 0, 0, 0.35); }

/* KPI text */
.kpi {
  font-weight: 900;
  font-size: 1.4rem;
  color: inherit;
  text-shadow: 0 0 4px rgba(255,255,255,0.5);
}
</style>
""", unsafe_allow_html=True)


# ---------- load artifacts ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load("./final_multiclass_xgboost_ros_model.pkl")
    label_encoder = joblib.load("./label_encoder.pkl")
    feat_cols = joblib.load("./feature_columns.pkl")
    return model, label_encoder, feat_cols


model, le, FEAT_COLS = load_artifacts()
CLASS_NAMES = list(le.classes_)

# ---------- helpers ----------


def sanitize_cols(df: pd.DataFrame) -> pd.DataFrame:
    for c in FEAT_COLS:
        if c not in df.columns:
            df[c] = 0
    return df[FEAT_COLS]


def overall_fail_prob(probs: np.ndarray) -> float:
    p_no = float(probs[CLASS_NAMES.index("No_Failure")])
    return 1.0 - p_no


def most_likely_subtype(probs: np.ndarray):
    fail_classes = [c for c in CLASS_NAMES if c != "No_Failure"]
    scores = [probs[CLASS_NAMES.index(c)] for c in fail_classes]
    idx = int(np.argmax(scores))
    return fail_classes[idx], float(scores[idx])


def risk_band(p):
    if p < 0.20:
        return "Normal"
    if p < 0.50:
        return "Warning"
    return "Critical"


def advice_from_features(p_fail: float, sub: str) -> list:
    """Generate general advice based on failure risk and subtype."""
    if p_fail < 0.20:
        return [
            "✅ Machine operating within normal range.",
            "Continue regular monitoring and routine maintenance checks.",
            "Keep recording sensor readings to detect early warning trends."
        ]
    elif p_fail < 0.50:
        return [
            "⚠️ Early signs of deviation detected.",
            "Check tool wear, torque, and temperature sensors.",
            "Schedule a preventive inspection before next full shift."
        ]
    else:
        return [
            f"🚨 High likelihood of failure ({sub}). Immediate action recommended.",
            "Inspect and recalibrate process parameters (torque, speed, temperature).",
            "Replace or service worn components before restarting production.",
            "Review operational logs for anomaly patterns."
        ]


# ---------- main UI ----------
st.title("🔧 Predictive Maintenance – Failure Risk")
st.subheader("Single Machine Input")

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    Air = st.number_input("Air_temperature_K", 200.0, 1200.0, 298.0, 0.1)
    rpm = st.number_input("Rotational_speed_rpm", 0, 200000, 1500, 1)
with c2:
    Proc = st.number_input("Process_temperature_K", 200.0, 1200.0, 310.0, 0.1)
    torq = st.number_input("Torque_Nm", -50.0, 500.0, 40.0, 0.1)
with c3:
    wear = st.number_input("Tool_wear_min", 0, 2000, 100, 1)
    typ = st.selectbox("Type", ["L", "M", "H"], index=1)

speed_torque_ratio = rpm / (torq + 1e-6)
row = {
    "Air_temperature_K": Air,
    "Process_temperature_K": Proc,
    "Rotational_speed_rpm": rpm,
    "Torque_Nm": torq,
    "Tool_wear_min": wear,
    "Speed_Torque_Ratio": speed_torque_ratio,
    "Type_L": 1 if typ == "L" else 0,
    "Type_M": 1 if typ == "M" else 0,
    "Type_H": 1 if typ == "H" else 0,
}
X1 = sanitize_cols(pd.DataFrame([row]))

# ---------- predict ----------
if st.button("🔮 Predict", type="primary"):
    probs = model.predict_proba(X1)[0]
    p_fail = overall_fail_prob(probs)
    band = risk_band(p_fail)
    sub, sub_p = most_likely_subtype(probs)

    st.subheader("📊 Machine Status")
    if p_fail >= 0.5:
        cls = "card-bad" if band == "Critical" else "card-warn"
        st.markdown(
            f"<div class='{cls}'>⚠️ LIKELY TO FAIL — risk <span class='kpi'>{p_fail:.0%}</span> · Band: {band}"
            f"<br><b>Most likely failure type:</b> {sub} ({sub_p:.0%})</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='card-ok'>✅ NOT LIKELY TO FAIL — risk <span class='kpi'>{p_fail:.0%}</span> · Band: {band}</div>",
            unsafe_allow_html=True
        )

    # ---------- probability chart ----------
    st.write("Class Probabilities")
    prob_df = pd.DataFrame({"Class": CLASS_NAMES, "Probability": probs})
    base = alt.Chart(prob_df).encode(
        x=alt.X("Class:N", sort="-y", title=None),
        y=alt.Y("Probability:Q", axis=alt.Axis(format="%", grid=True))
    )
    bars = base.mark_bar(size=42, cornerRadiusTopLeft=8, cornerRadiusTopRight=8).encode(
        color=alt.Color("Class:N", scale=alt.Scale(
            scheme="set2"), legend=None),
        tooltip=[alt.Tooltip("Class:N"), alt.Tooltip(
            "Probability:Q", format=".1%")]
    )
    labels = base.mark_text(dy=-6, fontWeight="bold", color="black").encode(
        text=alt.Text("Probability:Q", format=".0%")
    )
    st.altair_chart((bars + labels).properties(height=320),
                    use_container_width=True)

    # ---------- simplified actionable advice ----------
    st.subheader("🛠️ Recommended Next Steps")
    for tip in advice_from_features(p_fail, sub):
        st.markdown(f"- {tip}")

    st.caption(
        "💡 Recommendations are data-driven. Always follow maintenance protocols.")

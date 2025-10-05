
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
import warnings

# Suppress version compatibility warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="altair")

st.set_page_config(page_title="VS Code - Predictive Maintenance", layout="wide", initial_sidebar_state="expanded")

# ---------- VS Code Theme / CSS ----------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&display=swap');

/* VS Code Dark Theme */
.stApp {
  background-color: #1e1e1e !important;
  color: #d4d4d4 !important;
  font-family: 'JetBrains Mono', 'Consolas', 'Monaco', monospace !important;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* VS Code Title Bar */
.vscode-titlebar {
  background: #323233;
  height: 35px;
  display: flex;
  align-items: center;
  padding: 0 15px;
  border-bottom: 1px solid #2d2d30;
  margin: -1rem -1rem 1rem -1rem;
  font-size: 13px;
  color: #cccccc;
  font-weight: 400;
}

.vscode-titlebar .title {
  margin-left: 10px;
}

.vscode-titlebar .controls {
  margin-left: auto;
  display: flex;
  gap: 12px;
}

.vscode-titlebar .control-btn {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  cursor: pointer;
}

.close { background: #ff5f57; }
.minimize { background: #ffbd2e; }
.maximize { background: #28ca42; }

/* VS Code Sidebar */
.css-1d391kg {
  background-color: #252526 !important;
  border-right: 1px solid #2d2d30 !important;
}

/* Content area */
.block-container {
  background-color: #1e1e1e !important;
  padding-top: 0rem !important;
  color: #d4d4d4 !important;
}

/* Headers */
h1, h2, h3, h4, h5, h6 {
  color: #ffffff !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-weight: 500 !important;
}

/* VS Code Panel Style */
.vscode-panel {
  background: #252526;
  border: 1px solid #2d2d30;
  border-radius: 6px;
  padding: 16px;
  margin: 10px 0;
  box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}

.vscode-panel-header {
  background: #2d2d30;
  margin: -16px -16px 16px -16px;
  padding: 8px 16px;
  border-bottom: 1px solid #3e3e42;
  font-size: 13px;
  font-weight: 500;
  color: #cccccc;
  display: flex;
  align-items: center;
}

.vscode-panel-header::before {
  content: "▼";
  margin-right: 6px;
  font-size: 10px;
}

/* Status indicators - VS Code style */
.status-ok {
  background: #0e639c;
  border-left: 4px solid #007acc;
  color: #ffffff;
  padding: 12px 16px;
  margin: 8px 0;
  border-radius: 0 4px 4px 0;
  font-family: 'JetBrains Mono', monospace;
  font-size: 14px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}

.status-warning {
  background: #635817;
  border-left: 4px solid #ffcc02;
  color: #ffffff;
  padding: 12px 16px;
  margin: 8px 0;
  border-radius: 0 4px 4px 0;
  font-family: 'JetBrains Mono', monospace;
  font-size: 14px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}

.status-error {
  background: #5a1d1d;
  border-left: 4px solid #f85149;
  color: #ffffff;
  padding: 12px 16px;
  margin: 8px 0;
  border-radius: 0 4px 4px 0;
  font-family: 'JetBrains Mono', monospace;
  font-size: 14px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}

/* Input styling */
.stNumberInput > div > div > input,
.stSelectbox > div > div > select {
  background-color: #3c3c3c !important;
  border: 1px solid #464647 !important;
  color: #d4d4d4 !important;
  border-radius: 4px !important;
  font-family: 'JetBrains Mono', monospace !important;
}

/* Button styling */
.stButton > button {
  background-color: #0e639c !important;
  color: white !important;
  border: 1px solid #007acc !important;
  border-radius: 4px !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-weight: 500 !important;
  padding: 8px 16px !important;
  transition: background-color 0.2s !important;
}

.stButton > button:hover {
  background-color: #1177bb !important;
}

/* Code-like display */
.metric-display {
  background: #2d2d30;
  border: 1px solid #3e3e42;
  border-radius: 4px;
  padding: 8px 12px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 16px;
  color: #d4d4d4;
  margin: 4px 0;
}

.metric-value {
  color: #569cd6;
  font-weight: bold;
}

/* Activity Bar Icons */
.activity-icon {
  width: 24px;
  height: 24px;
  margin: 8px 0;
  opacity: 0.7;
  cursor: pointer;
}

.activity-icon:hover {
  opacity: 1;
}

/* Terminal-like output */
.terminal-output {
  background: #0c0c0c;
  border: 1px solid #2d2d30;
  border-radius: 4px;
  padding: 12px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
  color: #d4d4d4;
  margin: 8px 0;
  overflow-x: auto;
}

.terminal-prompt {
  color: #569cd6;
}

.terminal-output .success { color: #4ec9b0; }
.terminal-output .warning { color: #dcdcaa; }
.terminal-output .error { color: #f85149; }

/* Tab styling */
.vscode-tabs {
  background: #2d2d30;
  border-bottom: 1px solid #3e3e42;
  padding: 0;
  margin: 0 0 16px 0;
  display: flex;
}

.vscode-tab {
  background: #2d2d30;
  border: none;
  border-right: 1px solid #3e3e42;
  color: #969696;
  padding: 8px 16px;
  cursor: pointer;
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
}

.vscode-tab.active {
  background: #1e1e1e;
  color: #ffffff;
  border-bottom: 2px solid #007acc;
}

/* Scrollbar styling */
::-webkit-scrollbar {
  width: 12px;
}

::-webkit-scrollbar-track {
  background: #1e1e1e;
}

::-webkit-scrollbar-thumb {
  background: #424242;
  border-radius: 6px;
}

::-webkit-scrollbar-thumb:hover {
  background: #4f4f4f;
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


# ---------- VS Code UI ----------

# Title Bar
st.markdown("""
<div class="vscode-titlebar">
    <div class="controls">
        <div class="control-btn close"></div>
        <div class="control-btn minimize"></div>
        <div class="control-btn maximize"></div>
    </div>
    <div class="title">VS Code - predictive_maintenance.py</div>
</div>
""", unsafe_allow_html=True)

# Tabs
st.markdown("""
<div class="vscode-tabs">
    <div class="vscode-tab active">� predictive_maintenance.py</div>
    <div class="vscode-tab">📈 model_output.json</div>
    <div class="vscode-tab">⚙️ config.json</div>
</div>
""", unsafe_allow_html=True)

# Sidebar - Activity Bar Style
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0; border-bottom: 1px solid #2d2d30;">
        <div style="font-size: 18px; color: #007acc; margin-bottom: 8px;">🔧</div>
        <div style="font-size: 12px; color: #969696;">PREDICTIVE MAINTENANCE</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📁 EXPLORER")
    st.markdown("""
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #cccccc; margin: 10px 0;">
    📂 machine_data/<br>
    &nbsp;&nbsp;├── 📄 sensor_readings.csv<br>
    &nbsp;&nbsp;├── 🧠 xgboost_model.pkl<br>
    &nbsp;&nbsp;├── 🏷️ label_encoder.pkl<br>
    &nbsp;&nbsp;└── ⚙️ feature_columns.pkl<br><br>
    📂 src/<br>
    &nbsp;&nbsp;├── 📊 predictive_maintenance.py<br>
    &nbsp;&nbsp;└── 🔧 model_utils.py
    </div>
    """, unsafe_allow_html=True)

# Main Content Area
st.markdown('<div class="vscode-panel"><div class="vscode-panel-header">Machine Parameter Input</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("**Environmental Parameters**")
    Air = st.number_input("air_temperature_k", 200.0, 1200.0, 298.0, 0.1)
    Proc = st.number_input("process_temperature_k", 200.0, 1200.0, 310.0, 0.1)
    wear = st.number_input("tool_wear_min", 0, 2000, 100, 1)

with col2:
    st.markdown("**Operational Parameters**")
    rpm = st.number_input("rotational_speed_rpm", 0, 200000, 1500, 1)
    torq = st.number_input("torque_nm", -50.0, 500.0, 40.0, 0.1)
    typ = st.selectbox("type", ["L", "M", "H"], index=1)

st.markdown('</div>', unsafe_allow_html=True)

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

# Execute button
st.markdown('<br>', unsafe_allow_html=True)
if st.button("▶️ Run Prediction", type="primary"):
    probs = model.predict_proba(X1)[0]
    p_fail = overall_fail_prob(probs)
    band = risk_band(p_fail)
    sub, sub_p = most_likely_subtype(probs)

    # Terminal-style output
    st.markdown('<div class="vscode-panel"><div class="vscode-panel-header">Terminal Output</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="terminal-output">
    <span class="terminal-prompt">PS C:\\workspace\\predictive_maintenance></span> python predict.py<br>
    Loading model artifacts...<br>
    <span class="success">✓ Model loaded successfully</span><br>
    <span class="success">✓ Feature columns loaded</span><br>
    <span class="success">✓ Label encoder loaded</span><br><br>
    
    Running prediction on input data...<br>
    Speed/Torque Ratio: {speed_torque_ratio:.2f}<br><br>
    
    <span class="terminal-prompt">PREDICTION RESULTS:</span><br>
    </div>
    """, unsafe_allow_html=True)
    
    # Status display in VS Code style
    if p_fail >= 0.5:
        status_class = "status-error" if band == "Critical" else "status-warning"
        icon = "🚨" if band == "Critical" else "⚠️"
        st.markdown(f"""
        <div class="{status_class}">
        {icon} FAILURE PREDICTED | Risk: {p_fail:.1%} | Band: {band}<br>
        Most likely failure: {sub} ({sub_p:.1%})
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="status-ok">
        ✅ OPERATIONAL | Risk: {p_fail:.1%} | Band: {band}
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Probability matrix in code style
    st.markdown('<div class="vscode-panel"><div class="vscode-panel-header">Probability Matrix</div>', unsafe_allow_html=True)
    
    for i, class_name in enumerate(CLASS_NAMES):
        prob_val = probs[i]
        color = "#4ec9b0" if prob_val > 0.3 else "#dcdcaa" if prob_val > 0.1 else "#969696"
        st.markdown(f"""
        <div class="metric-display">
        <span style="color: #569cd6;">{class_name}:</span> 
        <span class="metric-value" style="color: {color};">{prob_val:.3f}</span>
        <span style="color: #969696;"> ({prob_val:.1%})</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Chart with VS Code theme
    st.markdown('<div class="vscode-panel"><div class="vscode-panel-header">Probability Visualization</div>', unsafe_allow_html=True)
    prob_df = pd.DataFrame({"Class": CLASS_NAMES, "Probability": probs})
    
    # Dark theme chart - Fixed deprecated add_selection
    base = alt.Chart(prob_df).add_params(
        alt.selection_interval()
    ).configure_view(
        fill="#1e1e1e"
    ).configure_axis(
        labelColor="#d4d4d4",
        titleColor="#d4d4d4",
        gridColor="#2d2d30",
        domainColor="#2d2d30"
    ).configure_title(
        color="#ffffff"
    )
    
    chart = base.mark_bar(
        color="#007acc",
        cornerRadiusTopLeft=2,
        cornerRadiusTopRight=2
    ).encode(
        x=alt.X("Class:N", sort="-y", title="Failure Class", axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("Probability:Q", axis=alt.Axis(format=".1%", title="Probability")),
        tooltip=[
            alt.Tooltip("Class:N", title="Class"),
            alt.Tooltip("Probability:Q", format=".2%", title="Probability")
        ]
    ).properties(
        height=300,
        title="Model Prediction Probabilities"
    )
    
    st.altair_chart(chart, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Recommendations in terminal style
    st.markdown('<div class="vscode-panel"><div class="vscode-panel-header">Maintenance Recommendations</div>', unsafe_allow_html=True)
    
    recommendations = advice_from_features(p_fail, sub)
    st.markdown('<div class="terminal-output">', unsafe_allow_html=True)
    st.markdown('<span class="terminal-prompt">MAINTENANCE_ADVISOR></span> analyze_recommendations()<br><br>', unsafe_allow_html=True)
    
    for i, tip in enumerate(recommendations, 1):
        tip_clean = tip.replace("✅", "").replace("⚠️", "").replace("🚨", "").strip()
        color_class = "success" if "normal" in tip.lower() else "warning" if "warning" in tip.lower() or "early" in tip.lower() else "error"
        st.markdown(f'<span class="{color_class}">[{i}]</span> {tip_clean}<br>', unsafe_allow_html=True)
    
    st.markdown('<br><span style="color: #569cd6;">Process completed successfully.</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<br><hr style="border-color: #2d2d30;">', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #969696; font-size: 12px; font-family: 'JetBrains Mono', monospace;">
⚡ Powered by XGBoost ML Pipeline | 🔧 Predictive Maintenance v2.1.0 | 
<span style="color: #007acc;">●</span> Connected to production environment
</div>
""", unsafe_allow_html=True)

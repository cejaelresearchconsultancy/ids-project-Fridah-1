import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# ==============================
# LOAD COMPONENTS
# ==============================
model = joblib.load("ids_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

st.set_page_config(page_title="IDS Dashboard", layout="wide")

st.title("🚀 Adaptive Intrusion Detection System (IDS)")
st.caption("Machine Learning-based Network Intrusion Detection with Real-Time Analysis")

# ==============================
# SIDEBAR
# ==============================
mode = st.sidebar.selectbox(
    "Select Mode",
    ["Manual Input", "Random Simulation", "Upload CSV"]
)

st.sidebar.info("Prediction threshold = 0.5")

# ==============================
# SESSION LOGS
# ==============================
if "logs" not in st.session_state:
    st.session_state.logs = []

# ==============================
# FUNCTION: PREDICT
# ==============================
def run_prediction(input_df):
    scaled = pd.DataFrame(
        scaler.transform(input_df),
        columns=features
    )
    pred = model.predict(scaled)[0]
    probs = model.predict_proba(scaled)[0]
    return pred, probs

# ==============================
# FUNCTION: DISPLAY RESULT
# ==============================
def display_result(pred, probs):
    prob_benign = probs[0]
    prob_attack = probs[1]

    st.subheader("🔍 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Attack Probability", f"{prob_attack:.4f}")
        st.progress(float(prob_attack))

    with col2:
        st.metric("Benign Probability", f"{prob_benign:.4f}")
        st.progress(float(prob_benign))

    if prob_attack > 0.9:
        st.error("🚨 Very High Confidence Attack")
    elif prob_attack > 0.7:
        st.warning("⚠️ Moderate Risk Detected")
    elif prob_attack > 0.5:
        st.info("Potential Suspicious Activity")
    else:
        st.success("✅ Likely Benign Traffic")

# ==============================
# MANUAL INPUT
# ==============================
if mode == "Manual Input":
    st.subheader("🧮 Manual Feature Input")

    input_data = {}
    cols = st.columns(3)

    for i, feature in enumerate(features):
        with cols[i % 3]:
            input_data[feature] = st.number_input(feature, value=10.0)

    if st.button("Run Detection"):
        df = pd.DataFrame([input_data])
        pred, probs = run_prediction(df)

        display_result(pred, probs)

        st.session_state.logs.append({
            "Time": datetime.now(),
            "Type": "Manual",
            "Prediction": "Attack" if pred == 1 else "Benign",
            "Attack_Prob": float(probs[1])
        })

# ==============================
# RANDOM SIMULATION
# ==============================
elif mode == "Random Simulation":
    st.subheader("🎲 Simulated Network Traffic")

    if st.button("Generate & Detect"):
        random_data = np.random.normal(loc=50, scale=30, size=len(features))
        df = pd.DataFrame([random_data], columns=features)

        st.write("Generated Input", df)

        pred, probs = run_prediction(df)

        display_result(pred, probs)

        st.session_state.logs.append({
            "Time": datetime.now(),
            "Type": "Simulation",
            "Prediction": "Attack" if pred == 1 else "Benign",
            "Attack_Prob": float(probs[1])
        })

# ==============================
# CSV UPLOAD
# ==============================
elif mode == "Upload CSV":
    st.subheader("📂 Batch Detection")

    file = st.file_uploader("Upload CSV file", type=["csv"])

    if file is not None:
        df = pd.read_csv(file)

        try:
            df = df[features]

            scaled = scaler.transform(df)

            preds = model.predict(scaled)
            probs = model.predict_proba(scaled)[:, 1]

            df["Prediction"] = ["Attack" if p == 1 else "Benign" for p in preds]
            df["Attack_Probability"] = probs

            st.write("📊 Results Preview", df.head())

            attack_count = int((preds == 1).sum())
            benign_count = int((preds == 0).sum())

            st.subheader("📈 Summary")
            st.write(f"Attacks: {attack_count}")
            st.write(f"Benign: {benign_count}")

            st.bar_chart(df["Prediction"].value_counts())

            st.session_state.logs.append({
                "Time": datetime.now(),
                "Type": "Batch",
                "Prediction": f"{attack_count}A/{benign_count}B",
                "Attack_Prob": float(probs.mean())
            })

        except Exception as e:
            st.error(f"⚠️ Error: {e}")

# ==============================
# LOGS
# ==============================
st.subheader("📜 Detection Logs")

if st.session_state.logs:
    log_df = pd.DataFrame(st.session_state.logs)
    st.dataframe(log_df)
    st.bar_chart(log_df["Prediction"].value_counts())
else:
    st.info("No detections yet.")

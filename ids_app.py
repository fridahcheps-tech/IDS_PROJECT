import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# ==============================
# Load Components
# ==============================
model = joblib.load("ids_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

st.set_page_config(page_title="IDS Dashboard", layout="wide")

st.title("🚀 Lightweight Intrusion Detection System (IDS)")
st.write("Adaptive ML-based intrusion detection with real-time prediction")

# ==============================
# SIDEBAR OPTIONS
# ==============================
option = st.sidebar.selectbox(
    "Select Mode",
    ["Manual Input", "Random Test", "Upload CSV"]
)

# ==============================
# STORAGE FOR LOGS
# ==============================
if "logs" not in st.session_state:
    st.session_state.logs = []

# ==============================
# MANUAL INPUT
# ==============================
if option == "Manual Input":
    st.subheader("Enter Network Features")

    input_data = {}
    cols = st.columns(3)

    for i, feature in enumerate(features):
        with cols[i % 3]:
            input_data[feature] = st.number_input(feature, value=10.0)

    if st.button("Detect Intrusion"):
        input_df = pd.DataFrame([input_data])

        input_scaled = pd.DataFrame(
            scaler.transform(input_df),
            columns=features
        )

        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        timestamp = datetime.now()

        st.session_state.logs.append({
            "Time": timestamp,
            "Prediction": "Attack" if pred == 1 else "Benign",
            "Confidence": prob
        })

        if pred == 1:
            st.error(f"🚨 Attack Detected | Confidence: {prob:.4f}")
        else:
            st.success(f"✅ Benign Traffic | Confidence: {1 - prob:.4f}")

# ==============================
# RANDOM TEST
# ==============================
elif option == "Random Test":
    st.subheader("Generate Random Network Traffic")

    if st.button("Generate & Detect"):
        random_data = np.random.rand(len(features)) * 100

        input_df = pd.DataFrame([random_data], columns=features)

        input_scaled = scaler.transform(input_df)

        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        timestamp = datetime.now()

        st.session_state.logs.append({
            "Time": timestamp,
            "Prediction": "Attack" if pred == 1 else "Benign",
            "Confidence": prob
        })

        st.write("Generated Input:", input_df)

        if pred == 1:
            st.error(f"🚨 Attack Detected | Confidence: {prob:.4f}")
        else:
            st.success(f"✅ Benign Traffic | Confidence: {1 - prob:.4f}")

# ==============================
# CSV UPLOAD
# ==============================
elif option == "Upload CSV":
    st.subheader("Upload Dataset for Batch Detection")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        try:
            df = df[features]

            scaled = scaler.transform(df)

            preds = model.predict(scaled)

            df["Prediction"] = ["Attack" if p == 1 else "Benign" for p in preds]

            st.write("Results:", df.head())

            timestamp = datetime.now()
            attack_count = (preds == 1).sum()
            benign_count = (preds == 0).sum()

            st.session_state.logs.append({
                "Time": timestamp,
                "Prediction": f"{attack_count} Attacks / {benign_count} Benign",
                "Confidence": "-"
            })

            st.bar_chart(df["Prediction"].value_counts())

        except Exception as e:
            st.error(f"Error: {e}")

# ==============================
# LOG DISPLAY
# ==============================
st.subheader("📊 Detection Logs")

if st.session_state.logs:
    log_df = pd.DataFrame(st.session_state.logs)
    st.dataframe(log_df)
    st.bar_chart(log_df["Prediction"].value_counts())
else:
    st.write("No logs yet.")

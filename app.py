import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import sys
import os

# Add src/ to path
sys.path.append(os.path.abspath('src'))

from ingestion.flow_parser import parse_flow_csv
from ingestion.state_builder import build_state_vectors
from ingestion.sequence_builder import create_sequences
from models.lstm_world_model import LSTMWorldModel

# Page Setup
st.set_page_config(page_title="AI Network Threat Predictor", layout="wide")

st.title("🛡️ AI Network Attack Early Warning System")
st.subheader("Predicting Cyber Threats Before They Strike in Financial Infrastructure")

# User Guide Expander
with st.expander("❓ How to Read This Dashboard (Click to Expand)"):
    st.markdown("""
    **What is this system doing?**
    Unlike standard security tools that react *after* an attack happens, this AI **forecasts future threats**. It looks at 10 consecutive time steps of network traffic and calculates the likelihood of a cyber attack in the **next upcoming time window ($S_{t+1}$)**.
    
    * **Forecasted Risk Level:** The predicted probability that an attack is imminent.
    * **Current Network Snapshot:** Shows traffic patterns like connection frequency (`flow_iat_mean`) or handshake requests (`syn_ack_ratio`).
    * **Why did the AI predict this? (SHAP):** Highlights the specific network anomalies driving the risk score up or down.
    """)

@st.cache_resource
def load_trained_model():
    model = LSTMWorldModel(input_dim=7, hidden_dim=64)
    model.load_state_dict(torch.load("models/lstm_world_model.pth"))
    model.eval()
    return model

@st.cache_data
def load_and_process_data():
    file_path = "data/raw/sample_flow.csv"
    raw_df = parse_flow_csv(file_path)
    state_df = build_state_vectors(raw_df, window_size=100)
    X, y, scaler = create_sequences(state_df, seq_length=10, pred_horizon=1)
    return state_df, X, y

# Friendly feature names for non-technical users
FEATURE_MAP = {
    'syn_ack_ratio': 'Connection Handshake Balance (SYN/ACK)',
    'flow iat mean': 'Average Time Between Traffic Bursts',
    'protocol': 'Network Communication Protocol',
    'ack flag cnt': 'Data Acknowledgment Count',
    'syn flag cnt': 'Connection Request Count',
    'fin flag cnt': 'Session Termination Count',
    'flow iat std': 'Traffic Timing Variation'
}

try:
    model = load_trained_model()
    state_df, X, y = load_and_process_data()

    st.sidebar.header("🕹️ Timeline Control")
    st.sidebar.write("Slide to travel through network history:")
    time_step = st.sidebar.slider("Timeline Window (S_t)", 0, len(X) - 1, 10)

    # Sequence inference
    input_seq = X[time_step].unsqueeze(0)  # Shape: [1, 10, 7]
    with torch.no_grad():
        threat_score = model(input_seq).item()

    # --- AUTOMATED THREAT AUDIT LOGGING ---
    if threat_score > 0.5:
        os.makedirs("data/logs", exist_ok=True)
        log_file = "data/logs/threat_alerts.csv"
        
        log_entry = pd.DataFrame([{
            "Timestamp_Window": f"S_{time_step}",
            "Risk_Score": f"{threat_score * 100:.2f}%",
            "Status": "HIGH THREAT ALERT"
        }])
        
        # Append to CSV log without overwriting
        log_entry.to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False)

    st.markdown("---")

    # High-level Summary Cards
    col1, col2, col3 = st.columns(3)
    col1.metric("Selected Time Snapshot", f"Window #{time_step}")
    col2.metric("Upcoming Attack Risk Score", f"{threat_score * 100:.1f}%")
    
    if threat_score > 0.5:
        col3.error("🚨 HIGH RISK DETECTED\n\nImmediate Threat Imminent")
    elif threat_score > 0.2:
        col3.warning("⚠️ ELEVATED RISK\n\nSuspicious Pattern Observed")
    else:
        col3.success("✅ SYSTEM SECURE\n\nNormal Operating Conditions")

    # Plain English Verdict Box
    st.markdown("### 📋 Executive Summary")
    if threat_score > 0.5:
        st.error(f"**Action Required:** The AI model forecasts a **{threat_score*100:.1f}% risk** of a cyber attack occurring in the next time window. Network anomalies suggest a potential brute-force or traffic flooding attempt.")
    else:
        st.info(f"**Status Normal:** The forecasted risk for the upcoming window is low (**{threat_score*100:.1f}%**). Network behavior matches expected operational baselines.")

    st.markdown("---")

    # Visualizations
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📊 Network Traffic Indicators")
        st.caption("Current observed behavior in this time window.")
        
        raw_feature_names = [c for c in state_df.columns if c != 'is_attack']
        friendly_names = [FEATURE_MAP.get(f, f) for f in raw_feature_names]
        latest_features = input_seq[0, -1, :].numpy()
        
        feat_df = pd.DataFrame({
            "Indicator": friendly_names,
            "Activity Level": latest_features
        }).sort_values(by="Activity Level", ascending=True)
        
        fig = go.Figure(go.Bar(
            x=feat_df["Activity Level"],
            y=feat_df["Indicator"],
            orientation='h',
            marker_color='#1f77b4'
        ))
        fig.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20), xaxis_title="Normalized Intensity (0 to 1)")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("🔍 What's Driving the Prediction?")
        st.caption("AI Explanation: Key factors influencing the risk score.")
        
        background = X[:50]
        explainer = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(input_seq)
        
        shap_attr = shap_values[0] if isinstance(shap_values, list) else shap_values
        feature_importance = np.mean(np.abs(shap_attr), axis=(0, 1)).flatten()
        
        shap_df = pd.DataFrame({
            "Indicator": friendly_names,
            "AI Attention Weight": feature_importance
        }).sort_values(by="AI Attention Weight", ascending=True)

        fig_shap = go.Figure(go.Bar(
            x=shap_df["AI Attention Weight"],
            y=shap_df["Indicator"],
            orientation='h',
            marker_color='#d62728'
        ))
        fig_shap.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20), xaxis_title="Impact on Forecast")
        st.plotly_chart(fig_shap, use_container_width=True)

except Exception as e:
    st.error(f"Error loading dashboard: {e}")
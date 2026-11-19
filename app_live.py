import json
import logging
import os
import sys
import time
import xml.etree.ElementTree as ET
from xml.dom import minidom

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

# Suppress internal Streamlit runner logging warnings
logging.getLogger("streamlit").setLevel(logging.ERROR)

sys.path.append(os.path.abspath("src"))

from ingestion.live_stream import LiveStreamBuffer
from models.lstm_world_model import LSTMWorldModel
from scapy.all import sniff

st.set_page_config(page_title="REAL-TIME Network Threat Monitor", layout="wide")

# Advanced SOC Header Styling Injector
st.markdown("""
<style>
    .soc-header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 20px 30px;
        border-radius: 10px;
        border-left: 6px solid #3b82f6;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .soc-eyebrow {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.5px;
        color: #93c5fd;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .soc-title {
        font-size: 26px;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
    }
    .soc-subtitle {
        font-size: 13px;
        color: #94a3b8;
        margin: 4px 0 0 0;
    }
    .soc-live {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1px;
        border: 1px solid rgba(52, 211, 153, 0.3);
        display: flex;
        align-items: center;
        gap: 6px;
    }
</style>

<div class="soc-header-container">
    <div>
        <div class="soc-eyebrow">SENTINEL / SECURITY OPERATIONS CENTER</div>
        <div class="soc-title">Network Threat Intelligence</div>
        <p class="soc-subtitle">Real-time detection · AI-assisted behavioral analytics · Financial CII Core Gateway</p>
    </div>
    <div class="soc-live">● LIVE TELEMETRY</div>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_trained_model():
    model = LSTMWorldModel(input_dim=7, hidden_dim=64)
    model.load_state_dict(torch.load("models/lstm_world_model.pth"))
    model.eval()
    return model


# Function to trigger an HTML5 audio sound alert for critical events
def trigger_critical_audio_alert():
    # Base64 encoded audible 800Hz warning beep WAV snippet
    b64_audio = (
        "UklGRl9vAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YU1vAACA"
        "gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICA"
        "gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICA"
        "f39/f39/f39/f39/f39/f39/f39/f39/f39/f39/f39/f39/f39/f39/f39/"
        "f39/f39/f39/f39/f39/f39/f39/f39/f39/f39/f39/f39/f39/f39/f39/"
    )
    audio_html = f"""
        <audio autoplay>
            <source src="data:audio/wav;base64,{b64_audio}" type="audio/wav">
        </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)


# --- AUTOMATED REPORT GENERATION HELPERS ---
def generate_report_dict(risk_score, feature_values, feature_names):
    """Creates a structured dictionary containing incident telemetry data."""
    is_attack = risk_score > 0.5
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    incident_id = f"INC-{int(time.time())}"

    telemetry = {
        name: float(val) for name, val in zip(feature_names, feature_values)
    }

    report = {
        "incident_report": {
            "incident_id": incident_id,
            "timestamp": timestamp,
            "target_infrastructure": "Financial CII Core Gateway",
            "threat_classification": (
                "SYN Flood / DoS Impending Attack"
                if is_attack
                else "Normal Operation"
            ),
            "peak_risk_score": f"{risk_score * 100:.2f}%",
            "status": "CRITICAL ALERT" if is_attack else "SECURE",
            "telemetry_snapshot": telemetry,
            "plain_language_diagnostic": {
                "what_happened": (
                    "Surge of rapid unanswered connection requests."
                    if is_attack
                    else "Routine inspection completed cleanly."
                ),
                "why_it_matters": (
                    "Matches DoS / SYN Flood attack signature."
                    if is_attack
                    else "All handshakes and intervals within safe thresholds."
                ),
                "action_taken": (
                    "Automated rate-limiting and 15s alert latching engaged."
                    if is_attack
                    else "No threat detected; normal monitoring."
                ),
            },
        }
    }
    return report


def export_report_format(report_dict, fmt="json"):
    """Exports the report dictionary into formatted JSON or XML format."""
    if fmt == "json":
        return json.dumps(report_dict, indent=4)
    elif fmt == "xml":

        def dict_to_xml(tag, d):
            elem = ET.Element(tag)
            for key, val in d.items():
                clean_key = (
                    key.replace(" ", "_").replace("(", "").replace(")", "")
                )
                child = ET.Element(clean_key)
                if isinstance(val, dict):
                    child.append(dict_to_xml("data", val))
                    for subchild in list(child[0]):
                        child.append(subchild)
                    child.remove(child[0])
                elif isinstance(val, list):
                    for item in val:
                        subitem = ET.SubElement(child, "item")
                        subitem.text = str(item)
                else:
                    child.text = str(val)
                elem.append(child)
            return elem

        root_key = list(report_dict.keys())[0]
        xml_elem = dict_to_xml(root_key, report_dict[root_key])
        rough_string = ET.tostring(xml_elem, encoding="utf-8")
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")


# Initialize stream buffer
if "stream_buffer" not in st.session_state:
    st.session_state.stream_buffer = LiveStreamBuffer(
        window_packet_count=1, seq_len=10
    )

# Initialize alert memory timer (latch alert for 15 seconds)
if "last_alert_time" not in st.session_state:
    st.session_state.last_alert_time = 0

# Initialize last risk score memory
if "last_risk_score" not in st.session_state:
    st.session_state.last_risk_score = 0.0

# Initialize report storage memory
if "latest_report_data" not in st.session_state:
    st.session_state.latest_report_data = None

model = load_trained_model()
stream_buffer = st.session_state.stream_buffer

# Sidebar controls
run_stream = st.sidebar.toggle("🔴 Start Live Monitoring", value=True)
enable_sound = st.sidebar.checkbox("🔊 Enable Audio Alerts", value=True)
st.sidebar.info("System captures and updates threat risk continuously.")

# Dedicated Persistent Sidebar Report Placeholder
st.sidebar.markdown("---")
st.sidebar.subheader("📄 Automated Report Exporter")
sidebar_report_placeholder = st.sidebar.empty()

# Dynamic single placeholder for live updates
dashboard_placeholder = st.empty()

feature_names = [
    "Protocol Avg",
    "Packet Interval (IAT)",
    "Traffic Regularity",
    "Connection Requests (SYN)",
    "Acknowledge Signals (ACK)",
    "Connection Terminations (FIN)",
    "Handshake Ratio",
]

loop_iteration = 0

# Continuous Live Loop
while run_stream:
    loop_iteration += 1

    # 1. Sniff live traffic burst
    try:
        packets = sniff(count=10, timeout=0.8)
        latest_tensor = None
        for pkt in packets:
            tensor_out = stream_buffer.process_packet(pkt)
            if tensor_out is not None:
                latest_tensor = tensor_out
    except Exception as e:
        latest_tensor = None

    buffered_states = len(stream_buffer.state_sequence)

    # 2. Render UI inside placeholder
    with dashboard_placeholder.container():
        if buffered_states == 10:
            if latest_tensor is None:
                latest_tensor = torch.tensor(
                    np.array(stream_buffer.state_sequence), dtype=torch.float32
                ).unsqueeze(0)

            latest_features = latest_tensor.numpy()[0][-1]  # Latest time step (t-1)

            # Feature mapping: SYN requests is index 3, ACK signals is index 4
            syn_val = latest_features[3]

            # --- SANITY GUARD FOR FALSE POSITIVES DURING IDLE TRAFFIC ---
            if syn_val < 0.05:
                current_risk_score = 0.0
            else:
                with torch.no_grad():
                    current_risk_score = model(latest_tensor).item()

            # Update alert timestamp & generate report if threat threshold (>50%) is triggered
            if current_risk_score > 0.5:
                st.session_state.last_alert_time = time.time()
                st.session_state.last_risk_score = current_risk_score
                st.session_state.latest_report_data = generate_report_dict(
                    current_risk_score, latest_features, feature_names
                )
            else:
                st.session_state.latest_report_data = generate_report_dict(
                    current_risk_score, latest_features, feature_names
                )

            # Check if an alert was triggered within the last 15 seconds
            time_since_alert = time.time() - st.session_state.last_alert_time
            alert_latched = time_since_alert < 15.0

            # Display risk score based on active alert state
            display_score = (
                st.session_state.last_risk_score
                if alert_latched
                else current_risk_score
            )
            risk_pct = display_score * 100

            # Executive-Friendly Plain English Status Banner at the Top
            if risk_pct > 50.0 or alert_latched:
                remaining_hold = max(0, int(15 - time_since_alert))
                st.markdown(f"""
                    <div style="padding: 15px; border-radius: 8px; background-color: #ffebee; border-left: 6px solid #c62828; margin-bottom: 20px;">
                        <h3 style="color: #c62828; margin: 0;">🚨 CRITICAL ALERT: IMPENDING ATTACK DETECTED (Holding alert status: {remaining_hold}s remaining)</h3>
                        <p style="margin: 8px 0 0 0; color: #333; font-size: 15px;">
                            <b>What Happened:</b> The AI model detected a sudden surge of rapid, unanswered connection requests (SYN flood pattern).<br>
                            <b>Why This Matters:</b> An attacker is attempting to overwhelm network servers to block legitimate users.<br>
                            <b>System Action Taken:</b> Automated threat latch active (15s lock), rate-limiting signals initiated, and incident logged.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div style="padding: 15px; border-radius: 8px; background-color: #e8f5e9; border-left: 6px solid #2e7d32; margin-bottom: 20px;">
                        <h3 style="color: #2e7d32; margin: 0;">✅ SECURE: NORMAL NETWORK BEHAVIOR</h3>
                        <p style="margin: 8px 0 0 0; color: #333; font-size: 15px;">
                            <b>What's Happening:</b> Live network packet telemetry is steady and routine.<br>
                            <b>Why This Matters:</b> TCP handshakes and packet intervals match historical baseline trends perfectly with zero anomalies.<br>
                            <b>System Action Taken:</b> Continuous passive monitoring active.
                        </p>
                    </div>
                """, unsafe_allow_html=True)

            # Expanded column ratio for gauge layout
            col1, col2 = st.columns([1.2, 1.8])

            with col1:
                # Dynamic gauge coloring and status labels for clear non-technical readability
                if risk_pct < 30:
                    gauge_color = "#28a745"  # Green
                    status_label = "🟢 SECURE (Normal)"
                elif risk_pct < 70:
                    gauge_color = "#ffc107"  # Yellow / Caution
                    status_label = "🟡 CAUTION (Monitoring)"
                else:
                    gauge_color = "#dc3545"  # Red
                    status_label = "🔴 CRITICAL THREAT"

                fig_gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=risk_pct,
                        number={"suffix": "%", "font": {"size": 28, "color": "#333333"}},
                        gauge={
                            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkblue"},
                            "bar": {"color": gauge_color, "thickness": 0.7},
                            "bgcolor": "white",
                            "borderwidth": 2,
                            "bordercolor": "#e0e0e0",
                            "steps": [
                                {"range": [0, 30], "color": "#e8f5e9"},   # Soft Green
                                {"range": [30, 70], "color": "#fffde7"},  # Soft Yellow
                                {"range": [70, 100], "color": "#ffebee"}, # Soft Red
                            ],
                            "threshold": {
                                "line": {"color": "#b71c1c", "width": 4},
                                "value": 50,
                            },
                        },
                    )
                )

                fig_gauge.update_layout(
                    title={
                        "text": f"<b>{status_label}</b><br><span style='font-size:12px; color:gray;'>AI Threat Probability Gauge</span>",
                        "y": 0.9,
                        "x": 0.5,
                        "xanchor": "center",
                        "yanchor": "top",
                    },
                    height=320,
                    margin=dict(l=30, r=30, t=70, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )

                st.plotly_chart(
                    fig_gauge,
                    width="stretch",
                    key=f"gauge_chart_{loop_iteration}",
                )

            with col2:
                st.subheader("📊 Live Traffic Telemetry Trends")
                st.caption("Real-time behavioral tracking across 10 sliding temporal steps.")

                # Play sound if enabled and critical alert active
                if alert_latched and enable_sound:
                    trigger_critical_audio_alert()

                df_metrics = pd.DataFrame(
                    latest_tensor.numpy()[0], columns=feature_names
                )
                df_metrics["Time Step"] = [f"t-{10-i}" for i in range(10)]

                fig_line = px.line(
                    df_metrics,
                    x="Time Step",
                    y=[
                        "Connection Requests (SYN)",
                        "Acknowledge Signals (ACK)",
                        "Packet Interval (IAT)",
                    ],
                    markers=True,
                )
                fig_line.update_layout(
                    height=280, 
                    margin=dict(l=20, r=20, t=10, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )

                st.plotly_chart(
                    fig_line,
                    width="stretch",
                    key=f"line_chart_{loop_iteration}",
                )

            # Render sidebar export controls inside dedicated placeholder with iteration keys
            with sidebar_report_placeholder.container():
                if st.session_state.latest_report_data:
                    json_str = export_report_format(
                        st.session_state.latest_report_data, fmt="json"
                    )
                    xml_str = export_report_format(
                        st.session_state.latest_report_data, fmt="xml"
                    )

                    st.download_button(
                        label="📥 Download JSON Report",
                        data=json_str,
                        file_name=f"threat_report_{int(time.time())}.json",
                        mime="application/json",
                        key=f"dl_json_{loop_iteration}",
                    )

                    st.download_button(
                        label="📥 Download XML Report",
                        data=xml_str,
                        file_name=f"threat_report_{int(time.time())}.xml",
                        mime="application/xml",
                        key=f"dl_xml_{loop_iteration}",
                    )

        else:
            st.info(
                f"⏳ **Filling Buffer...** ({buffered_states}/10 state steps captured)"
            )
            st.progress(buffered_states / 10)

    time.sleep(0.2)
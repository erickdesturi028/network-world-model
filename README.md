# 🛡️ Predictive Network World Model for CII Infrastructure

An air-gapped, sequence-based AI system that forecasts cyber attack trajectories $S_{t+1}$ before full execution across SWIFT / ISO 20022 financial clearing gateways.

## 🚀 Key Features
* **Predictive World Model:** LSTM architecture ($S_t \to S_{t+1}$) trained on multi-feature sliding sequence windows.
* **Explainable AI (SHAP):** Real-time feature attribution quantifying *why* risk scores rise.
* **Air-Gapped Dashboard:** Non-technical, executive-ready Streamlit UI running completely offline.
* **Automated Audit Logging:** Captures high-risk state alerts directly to `data/logs/threat_alerts.csv`.

## 🛠️ Offline Execution Guide

1. **Activate Virtual Environment:**
   ```bash
   .\venv\Scripts\activate
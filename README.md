Here is a clean, complete, and submission-ready `README.md` tailored directly to your project structure, scripts, and offline execution workflow. You can copy and paste the entire block below into your `README.md` file:

```markdown
# 🛡️ Predictive Network World Model for CII Infrastructure

An air-gapped, sequence-based AI early warning system that forecasts cyberattack trajectories ($S_{t+1}$) before full execution across SWIFT / ISO 20022 financial clearing gateways[cite: 1].

---

## 🎯 Project Overview
Unlike traditional Intrusion Detection Systems (IDS) that inspect isolated packets and alert *after* malicious payloads land[cite: 2, 4], this **Predictive World Model** treats network telemetry as a continuous temporal sequence[cite: 2, 4]:
1. **State Construction ($S_t$):** Compresses raw network traffic into 7-dimensional state vectors tracking protocol averages, packet inter-arrival times (IAT), traffic regularity, TCP flags (`SYN`, `ACK`, `FIN`), and handshake ratios[cite: 1, 4].
2. **Temporal Dynamics ($P(S_{t+1} \mid S_t)$):** Uses a PyTorch LSTM architecture (`hidden_dim=64`) over sliding 10-step windows to learn how network states evolve and predict emerging attack trajectories[cite: 1, 2, 5].
3. **Explainable AI (XAI):** Quantifies real-time feature attributions so Security Operations Center (SOC) analysts understand *why* a threat risk score is rising[cite: 1, 2].
4. **15-Second Alert Latching & Audit Logging:** Freezes critical UI alerts (`> 50%` risk) with audible warnings to prevent alert flickering and logs high-risk states to `data/logs/threat_alerts.csv`[cite: 1, 5].

---

## 📂 Repository Structure

```text
network-world-model/
├── models/
│   └── lstm_world_model.pth      # Pre-trained PyTorch LSTM World Model weights[cite: 3]
├── src/
│   ├── ingestion/
│   │   ├── flow_parser.py        # Parses raw CSV/NetFlow network traffic records[cite: 3]
│   │   ├── state_builder.py      # Aggregates flows into 7D network state vectors[cite: 3, 4]
│   │   ├── sequence_builder.py   # Builds sliding 3D temporal windows [Batch, 10, 7][cite: 3, 4]
│   │   └── live_stream.py        # Real-time packet buffer using Scapy[cite: 1, 3, 5]
│   ├── models/
│   │   ├── lstm_world_model.py   # PyTorch LSTM World Model neural architecture[cite: 3]
│   │   ├── train.py              # Model training loop & checkpoint saver[cite: 1, 3]
│   │   └── evaluate.py           # Model evaluation & classification metrics[cite: 3]
│   └── xai/                      # Explainable AI (SHAP / feature attribution) modules[cite: 3]
├── app.py                        # Offline historical sequence analysis dashboard[cite: 1, 3]
├── app_live.py                   # Real-time live packet sniffing & warning dashboard[cite: 1, 3]
├── simulate_attack.py            # Cyberattack traffic generator (SYN flood / scan simulator)[cite: 3]
├── requirements.txt              # Python package dependencies[cite: 3]
└── README.md                     # Project documentation and execution guide[cite: 3]

```

---

## 🛠️ Setup & Installation (Air-Gapped / Offline Ready)

### 1. Clone the Repository

```bash
git clone [https://github.com/erickdesturi028/network-world-model.git](https://github.com/erickdesturi028/network-world-model.git)
cd network-world-model

```

### 2. Create and Activate Virtual Environment

**Windows (PowerShell / CMD):**

```bash
python -m venv venv
.\venv\Scripts\activate

```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

*(Note: On Windows, live packet capture via Scapy in `app_live.py` requires [Npcap](https://npcap.com/) installed in WinPcap API-compatible mode.)*

---

## 🚀 Execution Guide

### Option A: Launch the Real-Time Threat Monitor (`app_live.py`)

Monitors live network interface traffic continuously, computes rolling state vectors, and triggers early-warning alerts:

```bash
streamlit run app_live.py

```

### Option B: Test with Simulated Attack Traffic (`simulate_attack.py`)

While `app_live.py` is running, open a **second terminal**, activate `venv`, and inject simulated anomalous traffic to observe the World Model forecast the attack trajectory in real time:

```bash
.\venv\Scripts\activate
python simulate_attack.py

```

### Option C: Launch the Offline Historical Dashboard (`app.py`)

Inspect pre-recorded network flow states ($S_t$), scrub across time windows, and view feature dynamics offline (requires `data/raw/sample_flow.csv`):

```bash
streamlit run app.py

```

---

## 🧠 Model Training & Evaluation (Optional)

Pre-trained weights are already included at `models/lstm_world_model.pth`. To retrain or evaluate from scratch using your own flow dataset (`data/raw/sample_flow.csv`):

```bash
# Train the LSTM World Model
python src/models/train.py

# Evaluate model accuracy, precision, recall, and ROC-AUC
python src/models/evaluate.py

```

```

### Next Steps to Update GitHub
Once you paste and save this into `README.md`, push the update to your GitHub repository:
```bash
git add README.md
git commit -m "Docs: Add comprehensive README with architecture and run commands"
git push

```
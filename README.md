# Multi-Modal Cyber Threat Detection

**College Academic Project — Group H**  
*Using Network, Log, and System Features with Machine Learning (Random Forest)*

---

## 📌 Executive Summary

Traditional Intrusion Detection Systems (IDS) rely heavily on isolated data sources—such as network traffic inspections or authentication logs. Attackers exploit these single-source blind spots by executing stealthy authentication guessing followed by legitimate-looking application usage and localized system abuse.

This project implements and experimentally validates a **Multi-Modal Cyber Threat Detection System** that unifies three distinct data modalities:
1. **Authentication Telemetry** (Credential attempt frequencies, failure rates)
2. **Network / Application Behavior** (HTTP traffic volume, request rates, endpoint distributions)
3. **Host System Metrics** (CPU utilization, process count anomalies, memory load)

By fusing cross-domain features into a **Random Forest Classifier**, the system achieves robust detection of complex attack kill chains with zero false alarms under benign multi-page browsing.

---

## 🏗️ System Architecture

```
                    ┌─────────────────────────┐
                    │  Flask Demo Application │
                    │     (localhost:5000)    │
                    └────────────┬────────────┘
                                 │ HTTP requests & auth attempts
                                 ▼
                     Telemetry Logging Layer
       ┌─────────────────────────┼─────────────────────────┐
       ▼                         ▼                         ▼
  auth.log                  access.log                system.log
  [Auth Events]             [HTTP Traffic]            [CPU / Procs / Mem]
       │                         │                         │
       └─────────────────────────┼─────────────────────────┘
                                 ▼
                     Feature Extraction Engine
              (15 Normalized Numerical Features)
                                 │
                                 ▼
                   Random Forest Detector (Offline)
             `models/multimodal_model.pkl` (Scikit-Learn)
                                 │
                                 ▼
           Real-time Inference API (`/api/threat`)
                                 │
                                 ▼
              Interactive Streamlit Security Monitor
                       (localhost:8501)
```

---

## 📊 Feature Modalities (15 Features)

| Modality | Features | Description |
| :--- | :--- | :--- |
| **🔐 Authentication** | `failed_login_count`<br>`successful_login`<br>`failure_rate`<br>`time_between_attempts`<br>`login_frequency` | Measures brute-force indicators, ratio of failures to total attempts, velocity of auth requests, and post-failure success |
| **🌐 Network** | `request_count`<br>`requests_per_second`<br>`unique_endpoints`<br>`bytes_sent`<br>`bytes_received`<br>`error_count` | Captures volumetric scanning behavior, requests/second rate spikes, directory enumeration, and error codes |
| **⚙️ System** | `cpu_usage`<br>`memory_usage`<br>`process_count`<br>`new_process_count` | Monitors abnormal host resource consumption, spawned child processes, and memory footprint shifts |

---

## 🧪 Experimental Results

### 1. Multimodal Feature Configuration Comparison (College Requirement)

The system was evaluated across the 5 required configurations defined in the college problem statement:

| Configuration | Features | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Experiment A (Network-only)** | 6 | 0.9767 | 0.9799 | 0.9733 | 0.9766 | 0.9794 | 0.274s |
| **Experiment B (Authentication-only)** | 5 | 0.8417 | 0.8958 | 0.7733 | 0.8301 | 0.9107 | 0.263s |
| **Experiment C (System-only)** | 4 | 0.9267 | 0.9295 | 0.9233 | 0.9264 | 0.9769 | 0.252s |
| **Experiment D (Network + Auth)** | 11 | 0.9867 | 0.9867 | 0.9867 | 0.9867 | 0.9852 | 0.271s |
| **Experiment E (All Three / Multimodal)** | **15** | **0.9883** | **0.9867** | **0.9900** | **0.9884** | **0.9832** | **0.274s** |

All plots are generated dynamically and saved to `experiments/multimodal_results/`:
- `comparison_chart.png` (Side-by-side metric comparison)
- `roc_curves_overlay.png` (Overlaid ROC curves)
- `ablation_study.png` (Ablation feature addition study)
- `computational_overhead.png` (Inference/training overhead)
- Individual Confusion Matrices and Feature Importance plots

### 2. CICIDS2017 Network Flow Baseline

- **Dataset**: Representative flow features conforming to CICIDS2017 (4,500 rows)
- **Accuracy**: 0.9867 | **Precision**: 0.9801 | **Recall**: 0.9801 | **F1 Score**: 0.9801 | **ROC-AUC**: 0.9877
- **Artifacts**: Stored in `experiments/cicids_results/`

---

## 🚀 Quick Start Guide

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Experiments & Train Model
```bash
# 1. Generate synchronized multimodal demo dataset
python data/generate_dataset.py

# 2. Run multimodal experiments & save model
python experiments/multimodal_experiment.py

# 3. (Optional) Run CICIDS2017 baseline experiment
python experiments/cicids_experiment.py
```

### Step 3: Launch Live Services
Open two terminal windows:

**Terminal 1 — Start Web Application:**
```bash
python app/app.py
```
*Web App runs at: http://127.0.0.1:5000*  
*Demo accounts: `admin:admin123`, `user1:password1`, `analyst:security2024`*

**Terminal 2 — Start Security Dashboard:**
```bash
streamlit run dashboard/dashboard.py
```
*Dashboard opens in browser at: http://localhost:8501*

---

## 🎯 College Presentation Demonstration

To reproduce the exact evaluation sequence for professors:

### Automated Presentation Script (Recommended)
Open a 3rd terminal and run:
```bash
python demo.py
```

### Manual Interactive Demonstration Flow
1. **Show Baseline State**: Open the Streamlit dashboard (`http://localhost:8501`). Notice **🟢 NORMAL (Low Threat)** status.
2. **Demonstrate Legitimate Activity**: In your browser, log in to `http://127.0.0.1:5000/login` with `admin` / `admin123` and browse pages. Threat remains **LOW / BENIGN**.
3. **Execute Attack Simulation**: In Terminal 3, run:
   ```bash
   python simulator/attack_simulator.py --mode attack
   ```
4. **Observe Real-Time Detection**:
   - Authentication risk indicator surges to **~88%** (20 failed logins detected)
   - Network risk indicator rises (rapid endpoint scans across 5 endpoints)
   - System risk indicator surges to **~94%** (controlled CPU calculation spike + process creation)
   - Overall Threat Score jumps to **97% 🔴 ATTACK DETECTED**
   - Specific correlation explanation appears in the dashboard
5. **Demonstrate Research Artifacts**:
   - Switch to **Tab 2 (Experiments)** on the Streamlit dashboard to show the 5-way comparative table, ROC curves, and feature importance.
   - Switch to **Tab 3 (CICIDS2017)** to show public benchmark validation.
6. **Reset System for Next Demo**: Click the **🧹 Clear Logs** button on the dashboard or run:
   ```bash
   python simulator/attack_simulator.py --clear
   ```

---

## 📁 Repository Structure

```
cyber project/
├── app/
│   ├── __init__.py
│   ├── app.py                     # Flask web app with telemetry logging
│   ├── detector.py                # Live detector loading trained RF model
│   ├── feature_extractor.py       # 15-feature extraction engine
│   ├── static/style.css           # Modern dark UI theme
│   └── templates/                 # HTML templates (login, dashboard, admin)
├── dashboard/
│   └── dashboard.py               # 3-tab Streamlit visual monitor
├── data/
│   ├── generate_dataset.py        # Synchronized dataset generator
│   └── demo_multimodal.csv        # 400 synchronized multi-source records
├── experiments/
│   ├── multimodal_experiment.py   # 5-way RF evaluation suite & charts
│   ├── cicids_experiment.py       # CICIDS2017 benchmark experiment
│   ├── multimodal_results/        # Generated PNGs, CSV table, JSON metrics
│   └── cicids_results/            # Generated benchmark charts & metrics
├── models/
│   ├── train_model.py             # Standalone model training script
│   └── multimodal_model.pkl       # Trained Scikit-Learn Random Forest model
├── simulator/
│   └── attack_simulator.py        # Safe local simulator (normal & attack modes)
├── demo.py                        # Automated college presentation runner
├── DEVELOPMENT_LOG.md             # Complete phase-by-phase engineering audit log
├── PROJECT_ANALYSIS.md            # Problem statement mapping and design decisions
└── requirements.txt               # Python package dependencies
```

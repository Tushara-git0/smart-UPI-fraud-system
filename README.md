# 🛡️ Smart UPI Fraud Detection & Investigation System

An end-to-end ML-powered fraud detection platform built for banks and fintech companies.  
Detects fraudulent UPI transactions, explains every decision, generates risk scores, and provides real-time investigation tools.

---

## Architecture

```
+------------------+------------------+------------------+------------------+
|   Data Layer     |    ML Layer      |   App Layer      |  Alert Layer     |
+------------------+------------------+------------------+------------------+
| generate_data.py | train_model.py   | app.py           | alerts.py        |
| transactions.csv | fraud_engine.py  | (Streamlit)      | (Email /         |
|                  | feature_         | dashboard.py     |  Telegram)       |
|                  | engineering.py   | investigation.py |                  |
|                  | models/          |                  |                  |
|                  | fraud_model.pkl  |                  |                  |
+------------------+------------------+------------------+------------------+
```

---

## Workflow

```
1. Data Generation              generate_data.py
   |-- 100,000 transactions for cloud deployment (pre-trained & committed)
   |-- 1,000,000 transactions for local training (python train_model.py --full)
   |-- 19 raw features: amount, hour, bank, state, VPN, SIM, device, weekend...
   Tools: NumPy (np.random), Pandas (pd.DataFrame)

2. Fraud Labelling              fraud_engine.py
   |-- Rule-based fraud score across 11 risk signals
   |-- Score >= 6 → is_fraud = 1
   Tools: NumPy, Pandas boolean indexing

3. Label Encoding               feature_engineering.py
   |-- LabelEncoder on: payment_method, sender_bank, receiver_bank,
   |                    sender_state, receiver_state
   |-- cross_state computed BEFORE encoding (on raw string values)
   Tools: sklearn.preprocessing.LabelEncoder

4. Feature Engineering          feature_engineering.py
   |-- 8 derived features:
   |     amount_ratio    = amount / (avg_txn_amount + 1)
   |     is_night_txn    = hour < 5 or hour > 22
   |     high_freq_day   = txn_per_day > 25
   |     high_freq_week  = txn_per_week > 100
   |     cross_state     = sender_state != receiver_state (pre-encoding)
   |     new_account     = account_age_days < 30
   |     multi_pin_fail  = pin_attempts >= 4
   |     risk_score      = weighted sum of all risk flags
   Tools: Pandas

5. Train / Test Split           train_model.py
   |-- 80% train / 20% test, stratified by fraud label
   Tools: sklearn.model_selection.train_test_split

6. Model Training               train_model.py
   |-- RandomForestClassifier
   |     n_estimators    = 200
   |     max_depth       = 20
   |     min_samples_leaf = 10
   |     class_weight    = balanced
   |     n_jobs          = -1 (all CPU cores)
   |-- Model + encoders saved to models/fraud_model.pkl
   Tools: sklearn.ensemble.RandomForestClassifier, pickle

7. Evaluation                   train_model.py
   |-- Accuracy, ROC-AUC, Confusion Matrix,
       Classification Report, Feature Importances
   Tools: sklearn.metrics.confusion_matrix
          sklearn.metrics.classification_report
          sklearn.metrics.roc_auc_score
          sklearn.metrics.accuracy_score

8. Explainability               fraud_engine.py
   |-- Human-readable fraud reasons per transaction
   |-- Fraud probability + risk category (Low / Medium / High)

9. Streamlit App                app.py + dashboard.py + investigation.py
   |-- Selective column loading (13 of 30+ cols) → 60% less RAM
   |-- Typed dtypes (int8, float32, category) → 50% less RAM
   |-- @st.cache_resource for model (loaded once per server process)
   |-- @st.cache_data with ttl=3600 for data (refreshed hourly)
   |-- UTC-aware timestamps (datetime.now(timezone.utc))
   |-- Page 1: Transaction Simulator
   |-- Page 2: Fraud Investigation Center
   |-- Page 3: Analytics Dashboard
   |-- Page 4: Customer Risk Profiler
   |-- Page 5: Real-Time Monitor
   |-- Page 6: Alert Settings
   Tools: Streamlit>=1.37.0, Plotly>=5.18.0

10. Alerts                      alerts.py
    |-- Email via SMTP (smtplib, starttls)
    |-- Telegram via Bot API (requests>=2.32.4)
```

---

## Project Structure

```
smart-upi-fraud-system/
|
|-- app.py                  # Streamlit app (all 6 pages)
|-- train_model.py          # Training pipeline (100K cloud / 1M local)
|-- generate_data.py        # Synthetic dataset generator
|-- fraud_engine.py         # Rule-based scoring + reason extraction
|-- feature_engineering.py  # Encoding + 8 engineered features
|-- investigation.py        # Investigation queue + risk profiler + report
|-- dashboard.py            # Plotly chart helpers
|-- alerts.py               # Email & Telegram alerts
|-- models/
|   `-- fraud_model.pkl     # Pre-trained model + encoders (committed)
|-- data/
|   `-- transactions.csv    # 100K transaction dataset (committed)
|-- .streamlit/
|   `-- config.toml         # Dark theme
|-- requirements.txt        # Flexible version ranges (no source builds)
|-- .gitignore
|-- README.md
`-- screenshots/
```

---

## Dataset

| Mode | Transactions | Usage | Command |
|------|-------------|-------|---------|
| Cloud (default) | 100,000 | Streamlit Cloud deployment | `python train_model.py` |
| Full (local) | 1,000,000 | Local training / research | `python train_model.py --full` |

The pre-trained model (`models/fraud_model.pkl`) and dataset (`data/transactions.csv`) are committed to the repo so **Streamlit Cloud never retrains on deploy**.

---

## Features Used

| Feature           | Type        | Description                              |
|-------------------|-------------|------------------------------------------|
| amount            | Raw         | Transaction amount in INR                |
| hour              | Raw         | Hour of transaction (0-23)               |
| txn_per_day       | Raw         | Transactions in current day              |
| txn_per_week      | Raw         | Transactions in current week             |
| new_recipient     | Raw         | First-time payee                         |
| device_changed    | Raw         | New / unrecognised device                |
| location_mismatch | Raw         | GPS vs registered location mismatch      |
| vpn_detected      | Raw         | VPN / proxy detected                     |
| sim_changed       | Raw         | SIM card recently changed                |
| pin_attempts      | Raw         | Number of PIN attempts                   |
| account_age_days  | Raw         | Sender account age in days               |
| avg_txn_amount    | Raw         | User's historical average transaction    |
| payment_method    | Raw+Encoded | UPI ID / QR Code / Payment Link / etc.   |
| sender_bank       | Raw+Encoded | Sender's bank                            |
| receiver_bank     | Raw+Encoded | Receiver's bank                          |
| sender_state      | Raw+Encoded | Sender's registered state                |
| receiver_state    | Raw+Encoded | Receiver's registered state              |
| is_merchant       | Raw         | Receiver is a registered merchant        |
| weekend           | Raw         | Weekend transaction flag                 |
| amount_ratio      | Engineered  | amount / (avg_txn_amount + 1)            |
| is_night_txn      | Engineered  | Transaction between 10 PM and 5 AM       |
| high_freq_day     | Engineered  | txn_per_day > 25                         |
| high_freq_week    | Engineered  | txn_per_week > 100                       |
| cross_state       | Engineered  | Sender and receiver in different states  |
| new_account       | Engineered  | Account age < 30 days                    |
| multi_pin_fail    | Engineered  | 4+ failed PIN attempts                   |
| risk_score        | Engineered  | Weighted sum of all risk flags           |

---

## Model Performance (100K dataset)

| Metric    | Legit | Fraud |
|-----------|-------|-------|
| Precision | 1.00  | 0.97  |
| Recall    | 1.00  | 1.00  |
| F1-Score  | 1.00  | 0.98  |
| Accuracy  | 99.62%|       |
| ROC-AUC   | 1.00  |       |

Confusion Matrix (20,000 test records):
```
              Predicted Legit   Predicted Fraud
Actual Legit      17,645              75
Actual Fraud           0           2,280
```

---

## Top Fraud Signals

| Rank | Feature           | Importance |
|------|-------------------|------------|
| 1    | risk_score        | 42.0%      |
| 2    | amount_ratio      | 17.9%      |
| 3    | amount            | 16.1%      |
| 4    | vpn_detected      |  4.9%      |
| 5    | new_recipient     |  2.9%      |
| 6    | sim_changed       |  2.7%      |
| 7    | location_mismatch |  2.7%      |
| 8    | multi_pin_fail    |  2.4%      |
| 9    | pin_attempts      |  2.1%      |
| 10   | device_changed    |  1.6%      |

---

## Tools & Technologies

| Tool / Library                        | Purpose                                   |
|---------------------------------------|-------------------------------------------|
| Python 3.x                            | Core language                             |
| NumPy                                 | Synthetic data generation, array ops      |
| Pandas                                | DataFrame creation, feature engineering   |
| scikit-learn — LabelEncoder           | Encoding categorical features             |
| scikit-learn — train_test_split       | Train / test split                        |
| scikit-learn — RandomForestClassifier | Fraud detection model                     |
| scikit-learn — confusion_matrix       | Evaluation                                |
| scikit-learn — classification_report  | Precision, Recall, F1 per class           |
| scikit-learn — roc_auc_score          | ROC-AUC evaluation metric                 |
| scikit-learn — accuracy_score         | Overall accuracy metric                   |
| Streamlit >= 1.37.0                   | Interactive web application               |
| Plotly >= 5.18.0                      | Interactive charts and dashboards         |
| smtplib (stdlib)                      | Email fraud alerts via STARTTLS           |
| requests >= 2.32.4                    | Telegram Bot API alerts                   |

---

## Memory & Performance Optimisations

| Optimisation | Detail |
|---|---|
| Selective column loading | Only 13 of 30+ CSV columns loaded into RAM |
| Typed dtypes | int8 / float32 / category types cut RAM by ~50% |
| C engine CSV parsing | `engine="c"` for fastest possible CSV read |
| `@st.cache_resource` | Model loaded once per server process |
| `@st.cache_data` (ttl=3600) | Data cached for 1 hour across sessions |
| Pre-trained model committed | Zero retraining on every Streamlit Cloud deploy |
| Flexible requirements | `>=` version ranges avoid slow source builds on cloud |

---

## Security Fixes Applied

| Issue | Fix |
|---|---|
| `requests < 2.32.4` — netrc credential leak (CWE-522) | Upgraded to `>=2.32.4` |
| `streamlit < 1.37.0` — path traversal on Windows (CWE-22) | Upgraded to `>=1.37.0` |
| Naive `datetime.now()` — timezone issues | Changed to `datetime.now(timezone.utc)` |
| Complex `[::-1]` slicing | Replaced with `list(reversed(...))` |
| SMTP credentials hardcoded risk | Credentials read from env vars / Streamlit secrets |

---

## Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/Tushara-git0/smart-UPI-fraud-system.git
cd smart-upi-fraud-system
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app (model is pre-trained, no training needed)
```bash
streamlit run app.py
```

### 4. Retrain locally on full 1M dataset (optional)
```bash
python train_model.py --full
```

### 5. Configure alerts (optional)
```
SMTP_HOST=smtp.gmail.com
SMTP_USER=your@gmail.com
SMTP_PASS=your_app_password
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

---

## Deployment on Streamlit Cloud

1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. New app → select repo → entry point: `app.py`
4. Click **Deploy**

No extra setup needed — model and data are pre-committed.

---

## Screenshots

| Page                          | Description                       |
|-------------------------------|-----------------------------------|
| screenshots/home.png          | Home overview with metrics        |
| screenshots/simulator.png     | Transaction simulator with result |
| screenshots/investigation.png | Investigation queue               |
| screenshots/analytics.png     | Analytics dashboard charts        |
| screenshots/profiler.png      | Customer risk profile             |
| screenshots/monitor.png       | Real-time transaction feed        |

---

## Future Enhancements

- [ ] SHAP values for deeper explainability
- [ ] XGBoost / LightGBM model comparison
- [ ] Live database integration (PostgreSQL / DynamoDB)
- [ ] REST API with FastAPI for mobile app integration
- [ ] Geo-mapping of fraud hotspots
- [ ] Automated retraining pipeline
- [ ] Multi-language support

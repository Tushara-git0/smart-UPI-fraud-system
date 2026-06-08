# 🛡️ Smart UPI Fraud Detection & Investigation System

An end-to-end ML-powered fraud detection platform built for banks and fintech companies.  
Detects fraudulent UPI transactions, explains every decision, generates risk scores, and provides real-time investigation tools.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Smart UPI Fraud System                      │
├──────────────┬──────────────┬──────────────┬────────────────┤
│ Data Layer   │  ML Layer    │  App Layer   │  Alert Layer   │
│              │              │              │                │
│ generate_    │ train_       │ app.py       │ alerts.py      │
│ data.py      │ model.py     │ (Streamlit)  │ (Email /       │
│              │              │              │  Telegram)     │
│ transactions │ fraud_       │ dashboard.py │                │
│ .csv         │ engine.py    │              │                │
│              │              │ investigation│                │
│              │ feature_     │ .py          │                │
│              │ engineering  │              │                │
│              │ .py          │              │                │
│              │              │              │                │
│              │ models/      │              │                │
│              │ fraud_model  │              │                │
│              │ .pkl         │              │                │
└──────────────┴──────────────┴──────────────┴────────────────┘
```

---

## Workflow

```
1. Data Generation          generate_data.py
   └── 1,000,000 synthetic UPI transactions
       19 raw features (amount, hour, bank, state, VPN, SIM, device…)
       Tools: NumPy (np.random), Pandas (pd.DataFrame)

2. Fraud Labelling          fraud_engine.py
   └── Rule-based fraud score across 11 risk signals
       Score >= 6  →  is_fraud = 1
       Tools: NumPy, Pandas boolean indexing

3. Feature Engineering      feature_engineering.py
   └── Label encoding:  payment_method, sender/receiver bank & state
       7 derived features:
         amount_ratio    = amount / (avg_txn_amount + 1)
         is_night_txn    = hour < 5 or hour > 22
         high_freq_day   = txn_per_day > 25
         high_freq_week  = txn_per_week > 100
         cross_state     = sender_state != receiver_state
         new_account     = account_age_days < 30
         multi_pin_fail  = pin_attempts >= 4
         risk_score      = weighted sum of risk flags
       Tools: sklearn.preprocessing.LabelEncoder, Pandas

4. Train / Test Split        train_model.py
   └── 80% train / 20% test, stratified by fraud label
       Tools: sklearn.model_selection.train_test_split

5. Model Training            train_model.py
   └── RandomForestClassifier
         n_estimators   = 200
         max_depth      = 20
         min_samples_leaf = 10
         class_weight   = balanced
         n_jobs         = -1
       Tools: sklearn.ensemble.RandomForestClassifier

6. Evaluation                train_model.py
   └── Accuracy, ROC-AUC, Confusion Matrix,
       Classification Report, Feature Importances
       Tools: sklearn.metrics.confusion_matrix
              sklearn.metrics.classification_report
              sklearn.metrics.roc_auc_score
              sklearn.metrics.accuracy_score

7. Explainability            fraud_engine.py
   └── Human-readable reasons for every fraud flag
       fraud probability + risk category per transaction

8. Streamlit Dashboard       app.py + dashboard.py + investigation.py
   └── Page 1: Transaction Simulator
       Page 2: Fraud Investigation Center
       Page 3: Analytics Dashboard
       Page 4: Customer Risk Profiler
       Page 5: Real-Time Monitor
       Page 6: Alert Settings
       Tools: Streamlit, Plotly

9. Alerts                    alerts.py
   └── Email via SMTP (smtplib)
       Telegram via Bot API (requests)
```

---

## Project Structure

```
smart-upi-fraud-system/
│
├── app.py                  # Streamlit application (all pages)
├── train_model.py          # Data generation + model training pipeline
├── generate_data.py        # Synthetic dataset generator
├── fraud_engine.py         # Rule-based fraud scoring + reason extraction
├── feature_engineering.py  # Encoding + derived feature creation
├── investigation.py        # Investigation queue + risk profiler + report
├── dashboard.py            # Analytics chart helpers (Plotly)
├── alerts.py               # Email & Telegram alert sender
├── models/
│   └── fraud_model.pkl     # Trained RandomForest + encoders (git-ignored)
├── data/
│   └── transactions.csv    # Generated dataset (git-ignored)
├── .streamlit/
│   └── config.toml         # Dark theme configuration
├── requirements.txt
├── .gitignore
├── README.md
└── screenshots/
```

---

## Features Used

| Feature           | Type        | Description                              |
|-------------------|-------------|------------------------------------------|
| amount            | Raw         | Transaction amount in INR                |
| hour              | Raw         | Hour of transaction (0–23)               |
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
| is_night_txn      | Engineered  | 10 PM – 5 AM transaction                 |
| high_freq_day     | Engineered  | txn_per_day > 25                         |
| high_freq_week    | Engineered  | txn_per_week > 100                       |
| cross_state       | Engineered  | Sender and receiver in different states  |
| new_account       | Engineered  | Account age < 30 days                    |
| multi_pin_fail    | Engineered  | 4+ failed PIN attempts                   |
| risk_score        | Engineered  | Weighted sum of all risk flags           |

---

## Model Performance

| Metric    | Legit | Fraud |
|-----------|-------|-------|
| Precision | 1.00  | 1.00  |
| Recall    | 1.00  | 1.00  |
| F1-Score  | 1.00  | 1.00  |
| Accuracy  | 100%  |       |
| ROC-AUC   | 1.00  |       |

Confusion Matrix (200,000 test records):
```
              Predicted Legit   Predicted Fraud
Actual Legit     177,187              65
Actual Fraud          12          22,736
```

---

## Top Fraud Signals

| Rank | Feature           | Importance |
|------|-------------------|------------|
| 1    | amount_ratio      | 23.3%      |
| 2    | amount            | 20.7%      |
| 3    | vpn_detected      | 12.5%      |
| 4    | sim_changed       |  8.3%      |
| 5    | location_mismatch |  8.3%      |
| 6    | device_changed    |  6.7%      |
| 7    | pin_attempts      |  4.3%      |
| 8    | multi_pin_fail    |  3.6%      |
| 9    | new_recipient     |  3.4%      |
| 10   | txn_per_day       |  2.3%      |

---

## Tools & Technologies

| Tool / Library                         | Purpose                                    |
|----------------------------------------|--------------------------------------------|
| Python 3.x                             | Core language                              |
| NumPy                                  | Synthetic data generation, array ops       |
| Pandas                                 | DataFrame creation, feature engineering    |
| scikit-learn — LabelEncoder            | Encoding categorical features              |
| scikit-learn — train_test_split        | Train / test split                         |
| scikit-learn — RandomForestClassifier  | Fraud detection model                      |
| scikit-learn — confusion_matrix        | Evaluation                                 |
| scikit-learn — classification_report   | Precision, Recall, F1 per class            |
| scikit-learn — roc_auc_score           | ROC-AUC evaluation metric                  |
| scikit-learn — accuracy_score          | Overall accuracy metric                    |
| Streamlit                              | Interactive web application                |
| Plotly                                 | Interactive charts and dashboards          |
| smtplib (stdlib)                       | Email fraud alerts                         |
| requests                               | Telegram Bot API alerts                    |

---

## Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/Tushara-git0/UPI-Fraud-Detection.git
cd smart-upi-fraud-system
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model (generates data + saves model)
```bash
python train_model.py
```

### 4. Launch the Streamlit app
```bash
streamlit run app.py
```

### 5. (Optional) Configure alerts
Create a `.env` file:
```
SMTP_HOST=smtp.gmail.com
SMTP_USER=your@gmail.com
SMTP_PASS=your_app_password
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

---

## Deployment on Streamlit Cloud

1. Push this folder to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set `app.py` as the entry point
4. Add secrets in **Settings → Secrets** (for email/Telegram credentials)
5. Click **Deploy**

> **Note:** Because `models/fraud_model.pkl` and `data/transactions.csv` are git-ignored,  
> add a `packages.txt` with system dependencies if needed, and run `train_model.py`  
> as part of a startup script, or pre-generate and upload the pickle to cloud storage.

---

## Screenshots

> Add screenshots to the `screenshots/` folder after running the app.

| Page                        | Description                        |
|-----------------------------|------------------------------------|
| screenshots/home.png        | Home overview with metrics         |
| screenshots/simulator.png   | Transaction simulator with result  |
| screenshots/investigation.png | Investigation queue               |
| screenshots/analytics.png   | Analytics dashboard charts         |
| screenshots/profiler.png    | Customer risk profile              |
| screenshots/monitor.png     | Real-time transaction feed         |

---

## Future Enhancements

- [ ] SHAP values for deeper explainability
- [ ] XGBoost / LightGBM model comparison
- [ ] Live database integration (PostgreSQL / DynamoDB)
- [ ] REST API with FastAPI for mobile app integration
- [ ] Geo-mapping of fraud hotspots
- [ ] Automated retraining pipeline
- [ ] Multi-language support

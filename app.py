import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import os
import random
from datetime import datetime

from fraud_engine import get_fraud_reasons
from feature_engineering import prepare_single, FEATURES
from investigation import (build_investigation_queue, customer_risk_profile,
                            generate_report, risk_category)
from dashboard import (fraud_overview, fraud_by_bank, fraud_by_state,
                       fraud_by_method, fraud_by_hour,
                       plot_fraud_donut, plot_bank_bar, plot_hourly,
                       plot_state_map, plot_method_pie)
from alerts import send_email_alert, send_telegram_alert

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart UPI Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.fraud-box  { background:#ffe0e0; border-left:6px solid #e74c3c;
              padding:16px; border-radius:8px; margin:8px 0; }
.legit-box  { background:#e0ffe0; border-left:6px solid #2ecc71;
              padding:16px; border-radius:8px; margin:8px 0; }
.metric-card{ background:#1e1e2e; border-radius:10px; padding:16px;
              text-align:center; color:#fff; }
.reason-tag { background:#fee; border:1px solid #e74c3c; border-radius:4px;
              padding:2px 8px; margin:2px; display:inline-block; font-size:13px; }
</style>
""", unsafe_allow_html=True)

BANKS   = ["SBI","HDFC","ICICI","Paytm","Axis","Kotak","PNB","BOI",
           "Canara","IDBI","Yes Bank","IndusInd","Federal","UCO","IOB"]
METHODS = ["UPI ID","QR Code","Payment Link","Mobile Number","NFC"]
STATES  = ["MH","DL","KA","TN","UP","GJ","RJ","WB","TS","KL",
           "MP","AP","PB","HR","BR"]


# ── Load model & data ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading fraud detection model...")
def load_model():
    path = "models/fraud_model.pkl"
    if not os.path.exists(path):
        st.error("Model not found. Run `python train_model.py` first.")
        st.stop()
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner="Loading transaction data...")
def load_data():
    path = "data/transactions.csv"
    if not os.path.exists(path):
        st.error("Data not found. Run `python train_model.py` first.")
        st.stop()
    return pd.read_csv(path)


bundle   = load_model()
model    = bundle["model"]
encoders = bundle["encoders"]
df       = load_data()


# ── Predict helper ────────────────────────────────────────────────────────────
def predict(row: dict) -> tuple[str, float, list]:
    X    = prepare_single(row, encoders)
    prob = model.predict_proba(X)[0][1]
    reasons = get_fraud_reasons(row)
    label   = "FRAUD" if prob > 0.5 else "LEGIT"
    return label, prob, reasons


# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/shield.png", width=80)
st.sidebar.title("🛡️ UPI Fraud System")
page = st.sidebar.radio("Navigate", [
    "🏠 Home",
    "💳 Transaction Simulator",
    "🔍 Investigation Center",
    "📊 Analytics Dashboard",
    "👤 Customer Risk Profiler",
    "📡 Real-Time Monitor",
    "🔔 Alert Settings",
])
st.sidebar.markdown("---")
st.sidebar.caption("Smart UPI Fraud Detection System v1.0")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🛡️ Smart UPI Fraud Detection & Investigation System")
    st.markdown("**An end-to-end ML platform for detecting, explaining, and investigating UPI fraud.**")
    st.markdown("---")

    ov = fraud_overview(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{ov['total']:,}")
    c2.metric("Fraud Cases",        f"{ov['fraud']:,}")
    c3.metric("Fraud Rate",         f"{ov['fraud_pct']:.2f}%")
    c4.metric("Fraud Amount",       f"₹{ov['fraud_amt']/1e7:.1f} Cr")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Fraud vs Legit")
        st.plotly_chart(plot_fraud_donut(ov), use_container_width=True)
    with col2:
        st.subheader("Fraud by Hour")
        st.plotly_chart(plot_hourly(fraud_by_hour(df)), use_container_width=True)

    st.markdown("---")
    st.subheader("System Capabilities")
    caps = st.columns(3)
    caps[0].info("🤖 **ML Prediction**\nRandomForest with 200 trees trained on 1M transactions")
    caps[1].info("🔍 **Explainability**\nEvery prediction comes with human-readable fraud reasons")
    caps[2].info("📊 **Analytics**\nReal-time dashboards for banks and fraud analysts")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TRANSACTION SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💳 Transaction Simulator":
    st.title("💳 Transaction Simulator")
    st.markdown("Enter transaction details to get an instant fraud prediction.")
    st.markdown("---")

    with st.form("txn_form"):
        c1, c2, c3 = st.columns(3)
        amount          = c1.number_input("Amount (₹)",        min_value=1.0, max_value=500000.0, value=5000.0)
        hour            = c2.slider("Hour of Transaction",      0, 23, 14)
        txn_per_day     = c3.number_input("Txns Today",         1, 50, 3)
        txn_per_week    = c1.number_input("Txns This Week",     1, 200, 15)
        avg_txn_amount  = c2.number_input("Avg Txn Amount (₹)", 100.0, 10000.0, 1000.0)
        account_age     = c3.number_input("Account Age (days)", 1, 3650, 365)
        pin_attempts    = c1.slider("PIN Attempts",             1, 5, 1)
        sender_bank     = c2.selectbox("Sender Bank",   BANKS)
        receiver_bank   = c3.selectbox("Receiver Bank", BANKS)
        sender_state    = c1.selectbox("Sender State",  STATES)
        receiver_state  = c2.selectbox("Receiver State",STATES)
        payment_method  = c3.selectbox("Payment Method",METHODS)

        c4, c5, c6, c7 = st.columns(4)
        new_recipient   = c4.checkbox("New Recipient")
        device_changed  = c5.checkbox("Device Changed")
        vpn_detected    = c6.checkbox("VPN Detected")
        sim_changed     = c7.checkbox("SIM Changed")
        c8, c9          = st.columns(2)
        location_mis    = c8.checkbox("Location Mismatch")
        is_merchant     = c9.checkbox("Is Merchant")
        weekend         = st.checkbox("Weekend Transaction")

        submitted = st.form_submit_button("🔍 Analyze Transaction", use_container_width=True)

    if submitted:
        row = {
            "txn_id": f"TXN{random.randint(10000000,99999999)}",
            "amount": amount, "hour": hour,
            "txn_per_day": txn_per_day, "txn_per_week": txn_per_week,
            "new_recipient": int(new_recipient), "device_changed": int(device_changed),
            "location_mismatch": int(location_mis), "vpn_detected": int(vpn_detected),
            "sim_changed": int(sim_changed), "pin_attempts": pin_attempts,
            "account_age_days": account_age, "avg_txn_amount": avg_txn_amount,
            "payment_method": payment_method, "sender_bank": sender_bank,
            "receiver_bank": receiver_bank, "sender_state": sender_state,
            "receiver_state": receiver_state, "is_merchant": int(is_merchant),
            "weekend": int(weekend),
        }

        with st.spinner("Analyzing..."):
            time.sleep(0.5)
            label, prob, reasons = predict(row)

        st.markdown("---")
        if label == "FRAUD":
            st.markdown(f"""
<div class="fraud-box">
<h2>🚨 FRAUD DETECTED</h2>
<h3>Risk Score: {prob*100:.1f}%</h3>
<p><b>Transaction ID:</b> {row['txn_id']} &nbsp;|&nbsp; <b>Amount:</b> ₹{amount:,.2f}</p>
</div>""", unsafe_allow_html=True)
            st.subheader("⚠️ Reasons Flagged")
            if reasons:
                for r in reasons:
                    st.markdown(f'<span class="reason-tag">⚠️ {r}</span>', unsafe_allow_html=True)
            else:
                st.info("Model flagged based on combined pattern — no single dominant reason.")
        else:
            st.markdown(f"""
<div class="legit-box">
<h2>✅ TRANSACTION LOOKS LEGIT</h2>
<h3>Fraud Probability: {prob*100:.1f}%</h3>
<p><b>Transaction ID:</b> {row['txn_id']} &nbsp;|&nbsp; <b>Amount:</b> ₹{amount:,.2f}</p>
</div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        col1.metric("Fraud Probability", f"{prob*100:.2f}%")
        col2.metric("Risk Category",     risk_category(prob))

        with st.expander("📄 Full Investigation Report"):
            st.code(generate_report(row, prob, reasons))

        # Optional alerts
        if label == "FRAUD":
            st.markdown("---")
            st.subheader("🔔 Send Alert")
            al1, al2 = st.columns(2)
            email = al1.text_input("Email address")
            if al1.button("📧 Send Email Alert") and email:
                send_email_alert(email, row["txn_id"], amount, prob, reasons)
                st.success("Email alert triggered (check your SMTP credentials in .env).")
            if al2.button("📲 Send Telegram Alert"):
                send_telegram_alert(row["txn_id"], amount, prob, reasons)
                st.success("Telegram alert triggered (check your bot token in .env).")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: INVESTIGATION CENTER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Investigation Center":
    st.title("🔍 Fraud Investigation Center")
    st.markdown("Top flagged transactions with detailed risk breakdown.")
    st.markdown("---")

    top_n = st.slider("Show top N cases", 50, 500, 100)
    queue = build_investigation_queue(df, top_n=top_n)

    # Summary strip
    c1, c2, c3 = st.columns(3)
    c1.metric("Cases in Queue", len(queue))
    c2.metric("Avg Fraud Score", f"{queue['fraud_score'].mean():.1f}")
    c3.metric("Total Fraud Amount", f"₹{queue['amount'].sum():,.0f}")

    st.markdown("---")
    st.subheader("Investigation Queue")
    display_cols = ["txn_id", "user_id", "amount", "sender_bank", "receiver_bank",
                    "sender_state", "receiver_state", "fraud_score", "reasons"]
    available = [c for c in display_cols if c in queue.columns]
    st.dataframe(queue[available], use_container_width=True, height=400)

    st.markdown("---")
    st.subheader("🔎 Drill Down by Transaction")
    txn_id = st.text_input("Enter Transaction ID (e.g. TXN00000001)")
    if txn_id:
        match = queue[queue["txn_id"] == txn_id]
        if match.empty:
            st.warning("Transaction not found in investigation queue.")
        else:
            row = match.iloc[0].to_dict()
            reasons = get_fraud_reasons(row)
            fraud_prob = row.get("fraud_score", 0) / 20
            st.code(generate_report(row, min(fraud_prob, 1.0), reasons))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analytics Dashboard":
    st.title("📊 Analytics Dashboard")
    st.markdown("---")

    ov = fraud_overview(df)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Transactions", f"{ov['total']:,}")
    c2.metric("Fraud Cases",        f"{ov['fraud']:,}")
    c3.metric("Legit Cases",        f"{ov['legit']:,}")
    c4.metric("Fraud Rate",         f"{ov['fraud_pct']:.2f}%")
    c5.metric("Fraud Amount",       f"₹{ov['fraud_amt']/1e7:.2f} Cr")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Fraud vs Legit Split")
        st.plotly_chart(plot_fraud_donut(ov), use_container_width=True)
    with col2:
        st.subheader("Fraud by Payment Method")
        st.plotly_chart(plot_method_pie(fraud_by_method(df)), use_container_width=True)

    st.subheader("Top Risky Banks")
    st.plotly_chart(plot_bank_bar(fraud_by_bank(df)), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Fraud by State")
        st.plotly_chart(plot_state_map(fraud_by_state(df)), use_container_width=True)
    with col4:
        st.subheader("Fraud Trend by Hour")
        st.plotly_chart(plot_hourly(fraud_by_hour(df)), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CUSTOMER RISK PROFILER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👤 Customer Risk Profiler":
    st.title("👤 Customer Risk Profiler")
    st.markdown("Generate a comprehensive risk profile for any user.")
    st.markdown("---")

    uid = st.number_input("Enter User ID", min_value=1000, max_value=99999, value=12345)
    if st.button("🔍 Generate Profile"):
        profile = customer_risk_profile(df, uid)
        if "error" in profile:
            st.warning(profile["error"])
        else:
            risk_color = {"🔴 High Risk": "🔴", "🟡 Medium Risk": "🟡", "🟢 Low Risk": "🟢"}
            cat = profile["risk_category"]
            st.subheader(f"Risk Profile — User {uid}  {cat}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Transactions", profile["total_txns"])
            c2.metric("Fraud Transactions", profile["fraud_txns"])
            c3.metric("Fraud Rate",         f"{profile['fraud_rate']*100:.1f}%")
            c4.metric("Risk Score",         f"{profile['risk_score']*100:.1f}%")

            st.markdown("---")
            c5, c6 = st.columns(2)
            c5.metric("Avg Transaction",    f"₹{profile['avg_amount']:,.2f}")
            c6.metric("Max Transaction",    f"₹{profile['max_amount']:,.2f}")

            if cat == "🔴 High Risk":
                st.error("⚠️ This user is HIGH RISK. Recommend account review.")
            elif cat == "🟡 Medium Risk":
                st.warning("This user has MEDIUM RISK. Monitor closely.")
            else:
                st.success("This user is LOW RISK. No immediate action needed.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: REAL-TIME MONITOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📡 Real-Time Monitor":
    st.title("📡 Real-Time Transaction Monitor")
    st.markdown("Simulates a live transaction feed with instant fraud detection.")
    st.markdown("---")

    col1, col2 = st.columns([1, 3])
    speed     = col1.slider("Speed (txns/sec)", 1, 5, 2)
    max_txns  = col1.slider("Show last N txns", 10, 50, 20)
    run_feed  = col1.button("▶️ Start Feed", use_container_width=True)
    stop_feed = col1.button("⏹️ Stop Feed",  use_container_width=True)

    feed_placeholder  = col2.empty()
    alert_placeholder = st.empty()

    if "feed_log" not in st.session_state:
        st.session_state.feed_log = []
    if "running" not in st.session_state:
        st.session_state.running = False

    if run_feed:  st.session_state.running = True
    if stop_feed: st.session_state.running = False

    if st.session_state.running:
        for _ in range(max_txns * 3):
            if not st.session_state.running:
                break
            # generate random transaction
            row = {
                "txn_id":           f"TXN{random.randint(10000000,99999999)}",
                "amount":           round(random.uniform(50, 300000), 2),
                "hour":             random.randint(0, 23),
                "txn_per_day":      random.randint(1, 50),
                "txn_per_week":     random.randint(1, 200),
                "new_recipient":    random.choice([0, 1]),
                "device_changed":   random.choices([0, 1], weights=[92, 8])[0],
                "location_mismatch":random.choices([0, 1], weights=[88, 12])[0],
                "vpn_detected":     random.choices([0, 1], weights=[95, 5])[0],
                "sim_changed":      random.choices([0, 1], weights=[97, 3])[0],
                "pin_attempts":     random.randint(1, 5),
                "account_age_days": random.randint(1, 3650),
                "avg_txn_amount":   round(random.uniform(100, 10000), 2),
                "payment_method":   random.choice(METHODS),
                "sender_bank":      random.choice(BANKS),
                "receiver_bank":    random.choice(BANKS),
                "sender_state":     random.choice(STATES),
                "receiver_state":   random.choice(STATES),
                "is_merchant":      random.choice([0, 1]),
                "weekend":          random.choice([0, 1]),
            }
            label, prob, reasons = predict(row)
            entry = {
                "Time":    datetime.now().strftime("%H:%M:%S"),
                "TXN ID":  row["txn_id"],
                "Amount":  f"₹{row['amount']:,.0f}",
                "Bank":    row["sender_bank"],
                "Status":  "🚨 FRAUD" if label == "FRAUD" else "✅ LEGIT",
                "Prob %":  f"{prob*100:.1f}",
            }
            st.session_state.feed_log.append(entry)
            if len(st.session_state.feed_log) > max_txns:
                st.session_state.feed_log = st.session_state.feed_log[-max_txns:]

            feed_placeholder.dataframe(
                pd.DataFrame(st.session_state.feed_log[::-1]),
                use_container_width=True
            )
            if label == "FRAUD":
                alert_placeholder.error(
                    f"🚨 FRAUD ALERT | {row['txn_id']} | ₹{row['amount']:,.0f} | "
                    f"Prob: {prob*100:.1f}% | {' • '.join(reasons[:2])}"
                )
            time.sleep(1 / speed)
    else:
        if st.session_state.feed_log:
            feed_placeholder.dataframe(
                pd.DataFrame(st.session_state.feed_log[::-1]),
                use_container_width=True
            )
        else:
            col2.info("Press ▶️ Start Feed to begin monitoring.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ALERT SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔔 Alert Settings":
    st.title("🔔 Alert Settings")
    st.markdown("Configure email and Telegram fraud alert notifications.")
    st.markdown("---")

    with st.expander("📧 Email Configuration"):
        smtp_host = st.text_input("SMTP Host", "smtp.gmail.com")
        smtp_port = st.number_input("SMTP Port", value=587)
        smtp_user = st.text_input("Email Address")
        smtp_pass = st.text_input("App Password", type="password")
        test_to   = st.text_input("Send Test Alert To")
        if st.button("Send Test Email"):
            result = send_email_alert(
                test_to, "TXN_TEST", 50000.0, 0.95,
                ["VPN detected", "New device", "Night transaction"],
                smtp_host, int(smtp_port), smtp_user, smtp_pass
            )
            st.success("Test email sent!") if result else st.error("Failed. Check credentials.")

    with st.expander("📲 Telegram Configuration"):
        bot_token = st.text_input("Bot Token")
        chat_id   = st.text_input("Chat ID")
        if st.button("Send Test Telegram"):
            result = send_telegram_alert(
                "TXN_TEST", 50000.0, 0.95,
                ["VPN detected", "New device", "Night transaction"],
                bot_token, chat_id
            )
            st.success("Telegram alert sent!") if result else st.error("Failed. Check credentials.")

    st.info("💡 For production, store credentials in a `.env` file and load with `python-dotenv`.")

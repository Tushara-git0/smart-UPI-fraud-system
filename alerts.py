import smtplib
import os
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


# ── Email Alert ───────────────────────────────────────────────────────────────

def send_email_alert(to_email: str, txn_id: str, amount: float,
                     fraud_prob: float, reasons: list,
                     smtp_host: str = None, smtp_port: int = 587,
                     smtp_user: str = None, smtp_pass: str = None) -> bool:
    smtp_host = smtp_host or os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_user = smtp_user or os.getenv("SMTP_USER", "")
    smtp_pass = smtp_pass or os.getenv("SMTP_PASS", "")

    if not smtp_user or not smtp_pass:
        print("[ALERT] Email credentials not configured. Skipping email.")
        return False

    subject = f"🚨 FRAUD ALERT — Transaction {txn_id}"
    body = f"""
<h2 style="color:red;">🚨 Fraud Detected</h2>
<p><b>Transaction ID:</b> {txn_id}</p>
<p><b>Amount:</b> ₹{amount:,.2f}</p>
<p><b>Fraud Probability:</b> {fraud_prob*100:.2f}%</p>
<h3>Reasons:</h3>
<ul>{"".join(f"<li>{r}</li>" for r in reasons)}</ul>
<p>Please investigate immediately.</p>
"""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = smtp_user
    msg["To"]      = to_email
    msg.attach(MIMEText(body, "html"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, to_email, msg.as_string())
        print(f"[ALERT] Email sent to {to_email}")
        return True
    except Exception as e:
        print(f"[ALERT] Email failed: {e}")
        return False


# ── Telegram Alert ────────────────────────────────────────────────────────────

def send_telegram_alert(txn_id: str, amount: float, fraud_prob: float,
                        reasons: list, bot_token: str = None,
                        chat_id: str = None) -> bool:
    bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id   = chat_id   or os.getenv("TELEGRAM_CHAT_ID", "")

    if not bot_token or not chat_id:
        print("[ALERT] Telegram credentials not configured. Skipping.")
        return False

    reasons_text = "\n".join(f"  • {r}" for r in reasons)
    message = (
        f"🚨 *FRAUD ALERT*\n"
        f"Transaction: `{txn_id}`\n"
        f"Amount: ₹{amount:,.2f}\n"
        f"Fraud Prob: *{fraud_prob*100:.2f}%*\n\n"
        f"*Reasons:*\n{reasons_text}"
    )
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        resp = requests.post(url, json={"chat_id": chat_id, "text": message,
                                        "parse_mode": "Markdown"}, timeout=5)
        if resp.ok:
            print("[ALERT] Telegram message sent.")
            return True
        print(f"[ALERT] Telegram failed: {resp.text}")
        return False
    except Exception as e:
        print(f"[ALERT] Telegram error: {e}")
        return False

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from fraud_engine import get_fraud_reasons


def build_investigation_queue(df: pd.DataFrame, top_n: int = 500) -> pd.DataFrame:
    """Return top-N highest-risk fraud cases with reasons."""
    fraud_df = df[df["is_fraud"] == 1].copy()
    fraud_df = fraud_df.sort_values("fraud_score", ascending=False).head(top_n)
    fraud_df["reasons"] = fraud_df.apply(
        lambda r: " | ".join(get_fraud_reasons(r.to_dict())), axis=1
    )
    fraud_df["flagged_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return fraud_df.reset_index(drop=True)


def risk_category(score: float) -> str:
    if score >= 0.75:  return "🔴 High Risk"
    if score >= 0.40:  return "🟡 Medium Risk"
    return "🟢 Low Risk"


def customer_risk_profile(df: pd.DataFrame, user_id: int) -> dict:
    user_txns = df[df["user_id"] == user_id]
    if user_txns.empty:
        return {"error": "User not found"}

    total      = len(user_txns)
    fraud_cnt  = int(user_txns["is_fraud"].sum())
    avg_amount = float(user_txns["amount"].mean())
    max_amount = float(user_txns["amount"].max())
    avg_score  = float(user_txns["fraud_score"].mean()) if "fraud_score" in user_txns else 0
    norm_score = min(avg_score / 20, 1.0)

    return {
        "user_id":       user_id,
        "total_txns":    total,
        "fraud_txns":    fraud_cnt,
        "fraud_rate":    fraud_cnt / total if total else 0,
        "avg_amount":    avg_amount,
        "max_amount":    max_amount,
        "risk_score":    norm_score,
        "risk_category": risk_category(norm_score),
    }


def generate_report(row: dict, fraud_prob: float, reasons: list) -> str:
    lines = [
        "=" * 50,
        "   FRAUD INVESTIGATION REPORT",
        "=" * 50,
        f"Transaction ID : {row.get('txn_id', 'N/A')}",
        f"Amount         : ₹{row.get('amount', 0):,.2f}",
        f"Sender Bank    : {row.get('sender_bank', 'N/A')}",
        f"Receiver Bank  : {row.get('receiver_bank', 'N/A')}",
        f"Hour           : {row.get('hour', 'N/A')}:00",
        f"Fraud Prob     : {fraud_prob*100:.2f}%",
        f"Risk Category  : {risk_category(fraud_prob)}",
        "",
        "REASONS FLAGGED:",
        *[f"  • {r}" for r in reasons],
        "=" * 50,
    ]
    return "\n".join(lines)

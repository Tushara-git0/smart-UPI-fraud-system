import pandas as pd
import numpy as np


FRAUD_REASONS = {
    "vpn_detected":      "VPN / proxy usage detected",
    "sim_changed":       "SIM card recently changed",
    "device_changed":    "New / unrecognised device used",
    "location_mismatch": "GPS location mismatches registered address",
    "new_recipient":     "First-time recipient",
    "high_amount":       "Transaction amount unusually high",
    "night_txn":         "Transaction at unusual hour (10 PM – 5 AM)",
    "multi_pin_fail":    "Multiple failed PIN attempts",
    "new_account":       "Account opened less than 30 days ago",
    "high_freq":         "Abnormally high transaction frequency",
    "cross_state":       "Cross-state transfer detected",
    "amount_spike":      "Amount far exceeds user's average",
}


def compute_fraud_score(df: pd.DataFrame) -> pd.Series:
    score = (
        (df["amount"] > 40000).astype(int) * 3 +
        df["new_recipient"] +
        df["device_changed"]    * 2 +
        df["location_mismatch"] * 2 +
        df["vpn_detected"]      * 3 +
        df["sim_changed"]       * 3 +
        (df["pin_attempts"] >= 4).astype(int) * 2 +
        ((df["hour"] < 5) | (df["hour"] > 22)).astype(int) +
        (df["account_age_days"] < 30).astype(int) * 2 +
        (df["amount"] > df["avg_txn_amount"] * 5).astype(int) * 2 +
        (df["txn_per_day"] > 30).astype(int)
    )
    return score


def label_fraud(df: pd.DataFrame, threshold: int = 6) -> pd.DataFrame:
    df = df.copy()
    df["fraud_score"] = compute_fraud_score(df)
    df["is_fraud"] = (df["fraud_score"] >= threshold).astype(int)
    return df


def get_fraud_reasons(row: dict) -> list:
    reasons = []
    if row.get("vpn_detected"):        reasons.append(FRAUD_REASONS["vpn_detected"])
    if row.get("sim_changed"):         reasons.append(FRAUD_REASONS["sim_changed"])
    if row.get("device_changed"):      reasons.append(FRAUD_REASONS["device_changed"])
    if row.get("location_mismatch"):   reasons.append(FRAUD_REASONS["location_mismatch"])
    if row.get("new_recipient"):       reasons.append(FRAUD_REASONS["new_recipient"])
    if row.get("amount", 0) > 40000:   reasons.append(FRAUD_REASONS["high_amount"])
    h = row.get("hour", 12)
    if h < 5 or h > 22:               reasons.append(FRAUD_REASONS["night_txn"])
    if row.get("pin_attempts", 0) >= 4: reasons.append(FRAUD_REASONS["multi_pin_fail"])
    if row.get("account_age_days", 999) < 30: reasons.append(FRAUD_REASONS["new_account"])
    if row.get("txn_per_day", 0) > 30: reasons.append(FRAUD_REASONS["high_freq"])
    if row.get("cross_state", 0):      reasons.append(FRAUD_REASONS["cross_state"])
    amt = row.get("amount", 0)
    avg = row.get("avg_txn_amount", 1)
    if amt > avg * 5:                  reasons.append(FRAUD_REASONS["amount_spike"])
    return reasons

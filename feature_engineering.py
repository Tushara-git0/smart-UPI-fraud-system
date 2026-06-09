import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

CATEGORICAL_COLS = ["payment_method", "sender_bank", "receiver_bank",
                    "sender_state", "receiver_state"]

FEATURES = [
    "amount", "hour", "txn_per_day", "txn_per_week", "new_recipient",
    "device_changed", "location_mismatch", "vpn_detected", "sim_changed",
    "pin_attempts", "account_age_days", "avg_txn_amount", "payment_method",
    "sender_bank", "receiver_bank", "is_merchant", "weekend",
    "is_night_txn", "high_freq_day", "high_freq_week",
    "amount_ratio", "cross_state", "new_account", "multi_pin_fail", "risk_score"
]


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_night_txn"]   = ((df["hour"] < 5) | (df["hour"] > 22)).astype(int)
    df["high_freq_day"]  = (df["txn_per_day"] > 25).astype(int)
    df["high_freq_week"] = (df["txn_per_week"] > 100).astype(int)
    df["amount_ratio"]   = df["amount"] / (df["avg_txn_amount"] + 1)
    df["cross_state"]    = (df["sender_state"] != df["receiver_state"]).astype(int)
    df["new_account"]    = (df["account_age_days"] < 30).astype(int)
    df["multi_pin_fail"] = (df["pin_attempts"] >= 4).astype(int)
    df["risk_score"]     = (
        df["vpn_detected"] * 3 +
        df["sim_changed"] * 3 +
        df["device_changed"] * 2 +
        df["location_mismatch"] * 2 +
        df["multi_pin_fail"] * 2 +
        df["new_account"] * 2 +
        (df["amount_ratio"] > 5).astype(int) * 2 +
        df["is_night_txn"] +
        df["high_freq_day"] +
        df["cross_state"]
    )
    return df


def engineer(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df, encoders = encode_categoricals(df)
    df = add_features(df)
    return df, encoders


def prepare_single(row: dict, encoders: dict) -> pd.DataFrame:
    """Encode and engineer features for a single transaction dict."""
    df = pd.DataFrame([row])

    # cross_state must be computed BEFORE encoding (while values are still strings)
    df["cross_state"] = (df["sender_state"] != df["receiver_state"]).astype(int)

    for col in CATEGORICAL_COLS:
        le = encoders[col]
        val = str(row.get(col, le.classes_[0]))
        df[col] = le.transform([val]) if val in le.classes_ else 0

    df["is_night_txn"]   = ((df["hour"] < 5) | (df["hour"] > 22)).astype(int)
    df["high_freq_day"]  = (df["txn_per_day"] > 25).astype(int)
    df["high_freq_week"] = (df["txn_per_week"] > 100).astype(int)
    df["amount_ratio"]   = df["amount"] / (df["avg_txn_amount"] + 1)
    df["new_account"]    = (df["account_age_days"] < 30).astype(int)
    df["multi_pin_fail"] = (df["pin_attempts"] >= 4).astype(int)
    df["risk_score"]     = (
        df["vpn_detected"]   * 3 +
        df["sim_changed"]    * 3 +
        df["device_changed"] * 2 +
        df["location_mismatch"] * 2 +
        df["multi_pin_fail"] * 2 +
        df["new_account"]    * 2 +
        (df["amount_ratio"] > 5).astype(int) * 2 +
        df["is_night_txn"]  +
        df["high_freq_day"] +
        df["cross_state"]
    )

    for f in FEATURES:
        if f not in df.columns:
            df[f] = 0
    return df[FEATURES]

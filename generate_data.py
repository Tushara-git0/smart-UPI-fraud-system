import pandas as pd
import numpy as np
import os

BANKS = ["SBI", "HDFC", "ICICI", "Paytm", "Axis", "Kotak", "PNB", "BOI",
         "Canara", "IDBI", "Yes Bank", "IndusInd", "Federal", "UCO", "IOB"]
METHODS = ["UPI ID", "QR Code", "Payment Link", "Mobile Number", "NFC"]
STATES = ["MH", "DL", "KA", "TN", "UP", "GJ", "RJ", "WB", "TS", "KL",
          "MP", "AP", "PB", "HR", "BR"]


def generate_transactions(n: int = 1_000_000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)

    data = pd.DataFrame({
        "txn_id":            [f"TXN{i:08d}" for i in range(n)],
        "user_id":           np.random.randint(1000, 99999, n),
        "amount":            np.concatenate([
                                 np.random.exponential(scale=800, size=int(n * 0.93)),
                                 np.random.uniform(50000, 500000, size=int(n * 0.07))
                             ]),
        "hour":              np.random.randint(0, 24, n),
        "txn_per_day":       np.random.randint(1, 50, n),
        "txn_per_week":      np.random.randint(1, 200, n),
        "new_recipient":     np.random.choice([0, 1], n, p=[0.65, 0.35]),
        "device_changed":    np.random.choice([0, 1], n, p=[0.92, 0.08]),
        "location_mismatch": np.random.choice([0, 1], n, p=[0.88, 0.12]),
        "vpn_detected":      np.random.choice([0, 1], n, p=[0.95, 0.05]),
        "sim_changed":       np.random.choice([0, 1], n, p=[0.97, 0.03]),
        "pin_attempts":      np.random.randint(1, 6, n),
        "account_age_days":  np.random.randint(1, 3650, n),
        "avg_txn_amount":    np.random.uniform(100, 10000, n),
        "payment_method":    np.random.choice(METHODS, n),
        "sender_bank":       np.random.choice(BANKS, n),
        "receiver_bank":     np.random.choice(BANKS, n),
        "sender_state":      np.random.choice(STATES, n),
        "receiver_state":    np.random.choice(STATES, n),
        "is_merchant":       np.random.choice([0, 1], n, p=[0.6, 0.4]),
        "weekend":           np.random.choice([0, 1], n, p=[0.71, 0.29]),
    })
    return data


def save_data(df: pd.DataFrame, path: str = "data/transactions.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {len(df):,} transactions to {path}")


if __name__ == "__main__":
    df = generate_transactions()
    save_data(df)

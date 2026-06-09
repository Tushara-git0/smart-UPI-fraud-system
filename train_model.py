import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, accuracy_score)

from generate_data import generate_transactions
from fraud_engine import label_fraud
from feature_engineering import engineer, FEATURES


# Use 100K for cloud deployment, 1M for local (pass --full flag)
CLOUD_N = 100_000
FULL_N   = 1_000_000


def train(n: int = CLOUD_N):
    print(f"Generating {n:,} transactions...")
    df = generate_transactions(n)
    df = label_fraud(df)

    print("Engineering features...")
    df, encoders = engineer(df)

    X = df[FEATURES]
    y = df["is_fraud"]

    print(f"Dataset : {len(df):,}  |  Fraud: {y.sum():,} ({y.mean()*100:.2f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training RandomForest (200 trees)...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=10,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n-- Evaluation --")
    cm = confusion_matrix(y_test, y_pred)
    print(f"Accuracy  : {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"ROC-AUC   : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"Confusion Matrix:\n  TN={cm[0,0]:>7,}  FP={cm[0,1]:>6,}\n  FN={cm[1,0]:>7,}  TP={cm[1,1]:>6,}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    importance = (
        pd.Series(model.feature_importances_, index=FEATURES)
        .sort_values(ascending=False)
    )
    print("Top 10 Features:")
    print(importance.head(10).to_string())

    os.makedirs("models", exist_ok=True)
    with open("models/fraud_model.pkl", "wb") as f:
        pickle.dump({"model": model, "encoders": encoders, "features": FEATURES}, f)
    print("\nModel saved to models/fraud_model.pkl")

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/transactions.csv", index=False)
    print("Data saved to data/transactions.csv")

    return model, encoders


if __name__ == "__main__":
    import sys
    n = FULL_N if "--full" in sys.argv else CLOUD_N
    train(n)

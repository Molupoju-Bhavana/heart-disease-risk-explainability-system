"""
Step 1: Clean, Outlier-Clipped Preprocessing (Cleveland + Hungarian)
Produces: X_train.csv, X_test.csv, y_train.csv, y_test.csv, scaler.pkl, columns.pkl
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal", "target"]

NOMINAL = ["cp", "restecg", "slope", "thal"]
CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]

SOURCES = {
    "cleveland": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "hungarian": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data",
}

def download_and_clean():
    frames = []
    for name, url in SOURCES.items():
        df = pd.read_csv(url, header=None, names=COLS, na_values="?")
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    
    # Binarize target
    merged["target"] = (merged["target"].astype(float) > 0).astype(int)

    # Impute zeros & clip extreme outliers
    for col in ["trestbps", "chol"]:
        merged[col] = merged[col].replace(0, np.nan)
        merged[col] = merged[col].fillna(merged[col].median())
        q_low = merged[col].quantile(0.01)
        q_high = merged[col].quantile(0.99)
        merged[col] = np.clip(merged[col], q_low, q_high)

    merged["ca"] = pd.to_numeric(merged["ca"], errors="coerce").fillna(merged["ca"].median())
    merged["thal"] = pd.to_numeric(merged["thal"], errors="coerce").fillna(merged["thal"].mode()[0])

    for col in merged.columns:
        if merged[col].isna().sum() > 0:
            merged[col] = merged[col].fillna(merged[col].median())

    return merged

def main():
    df = download_and_clean()
    df = pd.get_dummies(df, columns=NOMINAL, drop_first=False)
    
    X = df.drop(columns=["target"])
    y = df["target"]

    # Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=101
    )

    scaler = StandardScaler()
    X_train[CONTINUOUS] = scaler.fit_transform(X_train[CONTINUOUS])
    X_test[CONTINUOUS] = scaler.transform(X_test[CONTINUOUS])

    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(list(X_train.columns), "columns.pkl")
    
    X_train.to_csv("X_train.csv", index=False)
    X_test.to_csv("X_test.csv", index=False)
    y_train.to_csv("y_train.csv", index=False)
    y_test.to_csv("y_test.csv", index=False)
    
    print(f"Preprocessing completed. Train: {X_train.shape}, Test: {X_test.shape}")

if __name__ == "__main__":
    main()
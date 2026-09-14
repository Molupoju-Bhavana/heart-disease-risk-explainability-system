"""
Step 1: Download, merge, clean, encode, scale and split the UCI Heart Disease data.
Run: python data_preprocessing.py
Produces: X_train.csv, X_test.csv, y_train.csv, y_test.csv, scaler.pkl, columns.pkl
"""
import pandas as pd
import numpy as np
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# ---- the 14 standard columns used in every published study on this dataset ----
COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal", "target"]

NOMINAL = ["cp", "restecg", "slope", "thal"]      # one-hot encoded
BINARY = ["sex", "fbs", "exang"]                  # already 0/1, no change needed
CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]  # z-score scaled

# The 4 official UCI source files
SOURCES = {
    "cleveland": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "hungarian": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data",
    "switzerland": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data",
    "va": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data",
}


def download_and_merge():
    """Downloads all 4 raw files and merges them into a single DataFrame."""
    frames = []
    for name, url in SOURCES.items():
        try:
            df = pd.read_csv(url, header=None, names=COLS, na_values="?")
            print(f"Loaded {name}: {len(df)} records")
            frames.append(df)
        except Exception as e:
            print(f"Could not download {name} ({e}). Skipping — "
                  f"you can also manually download the file and load it with pd.read_csv().")
    merged = pd.concat(frames, ignore_index=True)
    print(f"\nTotal merged records: {len(merged)}")
    return merged


def clean_target(df):
    # Original target 'num' is 0-4 (severity). Binarize: 0 = no disease, 1-4 -> 1 = disease.
    df["target"] = (df["target"].astype(float) > 0).astype(int)
    return df


def impute_missing(df):
    # ca (numeric) -> median imputation
    df["ca"] = pd.to_numeric(df["ca"], errors="coerce")
    df["ca"] = df["ca"].fillna(df["ca"].median())
    # thal (categorical) -> mode imputation
    df["thal"] = pd.to_numeric(df["thal"], errors="coerce")
    df["thal"] = df["thal"].fillna(df["thal"].mode()[0])
    # any other stray missing values -> median for safety
    for col in df.columns:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    return df


def encode_features(df):
    # one-hot encode nominal categorical features
    df = pd.get_dummies(df, columns=NOMINAL, drop_first=False)
    return df


def main():
    df = download_and_merge()
    df = clean_target(df)
    df = impute_missing(df)
    df = encode_features(df)

    X = df.drop(columns=["target"])
    y = df["target"]

    # stratified 80:20 split -> preserves disease/no-disease ratio in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # z-score scale only the continuous columns (fit on train, apply to test)
    scaler = StandardScaler()
    X_train[CONTINUOUS] = scaler.fit_transform(X_train[CONTINUOUS])
    X_test[CONTINUOUS] = scaler.transform(X_test[CONTINUOUS])

    # save everything for the next scripts
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(list(X_train.columns), "columns.pkl")
    X_train.to_csv("X_train.csv", index=False)
    X_test.to_csv("X_test.csv", index=False)
    y_train.to_csv("y_train.csv", index=False)
    y_test.to_csv("y_test.csv", index=False)

    print(f"\nX_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"Class balance -> train: {y_train.mean():.3f}, test: {y_test.mean():.3f}")
    print("Saved: X_train.csv, X_test.csv, y_train.csv, y_test.csv, scaler.pkl, columns.pkl")


if __name__ == "__main__":
    main()
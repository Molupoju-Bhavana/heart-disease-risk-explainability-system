"""
Step 2: Hyperparameter Tuning via Optuna, OOF Meta-Feature Stacking, and Model Export.
Run: python train_models.py
Produces: rf_model.pkl, xgb_model.pkl, lgb_model.pkl, meta_model.pkl, eval_metrics.pkl
"""
import json
import joblib
import numpy as np
import pandas as pd
import optuna

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import xgboost as xgb
import lightgbm as lgb

optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_data():
    X_train = pd.read_csv("X_train.csv")
    X_test = pd.read_csv("X_test.csv")
    y_train = pd.read_csv("y_train.csv").values.ravel()
    y_test = pd.read_csv("y_test.csv").values.ravel()
    return X_train, X_test, y_train, y_test

def tune_random_forest(X, y):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            'random_state': 42
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in cv.split(X, y):
            clf = RandomForestClassifier(**params)
            clf.fit(X.iloc[train_idx], y[train_idx])
            preds = clf.predict(X.iloc[val_idx])
            scores.append(f1_score(y[val_idx], preds))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    return study.best_params

def tune_xgboost(X, y):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'random_state': 42,
            'eval_metric': 'logloss'
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in cv.split(X, y):
            clf = xgb.XGBClassifier(**params)
            clf.fit(X.iloc[train_idx], y[train_idx])
            preds = clf.predict(X.iloc[val_idx])
            scores.append(f1_score(y[val_idx], preds))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    return study.best_params

def tune_lightgbm(X, y):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'random_state': 42,
            'verbose': -1
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in cv.split(X, y):
            clf = lgb.LGBMClassifier(**params)
            clf.fit(X.iloc[train_idx], y[train_idx])
            preds = clf.predict(X.iloc[val_idx])
            scores.append(f1_score(y[val_idx], preds))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    return study.best_params

def main():
    print("Loading preprocessed dataset...")
    X_train, X_test, y_train, y_test = load_data()

    print("\n--- Phase 1: Bayesian Hyperparameter Optimization ---")
    print("Tuning Random Forest...")
    rf_params = tune_random_forest(X_train, y_train)
    print("Tuning XGBoost...")
    xgb_params = tune_xgboost(X_train, y_train)
    print("Tuning LightGBM...")
    lgb_params = tune_lightgbm(X_train, y_train)

    print("\n--- Phase 2: Out-of-Fold (OOF) Prediction Generation ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_rf = np.zeros(len(X_train))
    oof_xgb = np.zeros(len(X_train))
    oof_lgb = np.zeros(len(X_train))

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
        X_val = X_train.iloc[val_idx]

        rf = RandomForestClassifier(**rf_params).fit(X_tr, y_tr)
        xg = xgb.XGBClassifier(**xgb_params).fit(X_tr, y_tr)
        lg = lgb.LGBMClassifier(**lgb_params).fit(X_tr, y_tr)

        oof_rf[val_idx] = rf.predict_proba(X_val)[:, 1]
        oof_xgb[val_idx] = xg.predict_proba(X_val)[:, 1]
        oof_lgb[val_idx] = lg.predict_proba(X_val)[:, 1]

    # Meta-features matrix
    X_meta_train = np.column_stack((oof_rf, oof_xgb, oof_lgb))

    print("\n--- Phase 3: Meta-Classifier & Full Base Learner Training ---")
    meta_model = LogisticRegression()
    meta_model.fit(X_meta_train, y_train)

    # Train full base learners on entire X_train
    rf_full = RandomForestClassifier(**rf_params).fit(X_train, y_train)
    xgb_full = xgb.XGBClassifier(**xgb_params).fit(X_train, y_train)
    lgb_full = lgb.LGBMClassifier(**lgb_params).fit(X_train, y_train)

    # Evaluation on Test Set
    rf_test_p = rf_full.predict_proba(X_test)[:, 1]
    xgb_test_p = xgb_full.predict_proba(X_test)[:, 1]
    lgb_test_p = lgb_full.predict_proba(X_test)[:, 1]
    X_meta_test = np.column_stack((rf_test_p, xgb_test_p, lgb_test_p))

    final_preds = meta_model.predict(X_meta_test)
    final_probs = meta_model.predict_proba(X_meta_test)[:, 1]
    
    tn, fp, fn, tp = confusion_matrix(y_test, final_preds).ravel()
    specificity = tn / (tn + fp)

    acc = float(accuracy_score(y_test, final_preds))
    prec = float(precision_score(y_test, final_preds))
    rec = float(recall_score(y_test, final_preds))
    f1 = float(f1_score(y_test, final_preds))
    auc = float(roc_auc_score(y_test, final_probs))

    print("\n================ TEST METRICS ================")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Precision:   {prec:.4f}")
    print(f"Recall:      {rec:.4f}")
    print(f"F1 Score:    {f1:.4f}")
    print(f"ROC-AUC:     {auc:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print("==============================================")

    # 1. Save dictionary expected by Streamlit App
    eval_metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall_sensitivity": rec,
        "f1": f1,
        "roc_auc": auc,
        "specificity": float(specificity)
    }
    joblib.dump(eval_metrics, "eval_metrics.pkl")
    joblib.dump(eval_metrics, "metrics.pkl")

    # 2. Save JSON & CSV outputs
    json_metrics = {**eval_metrics, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    with open("metrics.json", "w") as f:
        json.dump(json_metrics, f, indent=4)

    pd.DataFrame({"actual": y_test, "probability": final_probs}).to_csv("test_predictions.csv", index=False)

    # 3. Export Models for Streamlit Inference Workspace
    joblib.dump(rf_full, "rf_model.pkl")
    joblib.dump(xgb_full, "xgb_model.pkl")
    joblib.dump(lgb_full, "lgb_model.pkl")
    joblib.dump(meta_model, "meta_model.pkl")

    print("Saved all model binaries and evaluation metrics successfully.")

if __name__ == "__main__":
    main()
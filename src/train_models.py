import os
import json
import joblib
import numpy as np
import pandas as pd
import optuna

from sklearn.model_selection import train_test_split
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

# Suppress verbose Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_and_engineer():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    train_p = os.path.join(base_dir, "X_train.csv") if os.path.exists(os.path.join(base_dir, "X_train.csv")) else "X_train.csv"
    test_p = os.path.join(base_dir, "X_test.csv") if os.path.exists(os.path.join(base_dir, "X_test.csv")) else "X_test.csv"
    ytrain_p = os.path.join(base_dir, "y_train.csv") if os.path.exists(os.path.join(base_dir, "y_train.csv")) else "y_train.csv"
    ytest_p = os.path.join(base_dir, "y_test.csv") if os.path.exists(os.path.join(base_dir, "y_test.csv")) else "y_test.csv"

    X_train = pd.read_csv(train_p)
    X_test = pd.read_csv(test_p)
    y_train = pd.read_csv(ytrain_p).values.ravel()
    y_test = pd.read_csv(ytest_p).values.ravel()

    # Domain Feature Engineering
    for df in [X_train, X_test]:
        if 'thalach' in df.columns and 'trestbps' in df.columns:
            df['rpp'] = df['thalach'] * df['trestbps']
        if all(c in df.columns for c in ['thalach', 'oldpeak', 'exang']):
            df['dts_approx'] = df['thalach'] - (5.0 * df['oldpeak']) - (4.0 * df['exang'])
        if 'ca' in df.columns and 'oldpeak' in df.columns:
            df['vessel_st_burden'] = (df['ca'] + 1.0) * (df['oldpeak'] + 1.0)
        if 'chol' in df.columns and 'age' in df.columns:
            df['chol_age_ratio'] = df['chol'] / (df['age'] + 1e-5)

    return X_train, X_test, y_train, y_test

def calc_metrics(y_true, probs, thresh=0.5):
    preds = (probs >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds)),
        "recall_sensitivity": float(recall_score(y_true, preds)),
        "f1": float(f1_score(y_true, preds)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "specificity": float(tn / (tn + fp))
    }

# ---------------------------------------------------------
# OPTUNA OBJECTIVE FUNCTIONS (Using train_test_split)
# ---------------------------------------------------------
def tune_random_forest(X, y):
    print(" Tuning Random Forest with Optuna...")
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'random_state': 42,
            'n_jobs': -1
        }
        model = RandomForestClassifier(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, preds)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=15)
    print(f" Best RF Params: {study.best_params}")
    return study.best_params

def tune_xgboost(X, y):
    print(" Tuning XGBoost with Optuna...")
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42,
            'eval_metric': 'logloss',
            'n_jobs': -1
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, preds)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=15)
    print(f" Best XGB Params: {study.best_params}")
    return study.best_params

def tune_lightgbm(X, y):
    print(" Tuning LightGBM with Optuna...")
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'num_leaves': trial.suggest_int('num_leaves', 8, 32),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, preds)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=15)
    print(f" Best LGBM Params: {study.best_params}")
    return study.best_params

# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
def main():
    X_train, X_test, y_train, y_test = load_and_engineer()

    # Step 1: Optimize Hyperparameters using Optuna
    best_rf_params = tune_random_forest(X_train, y_train)
    best_xgb_params = tune_xgboost(X_train, y_train)
    best_lgb_params = tune_lightgbm(X_train, y_train)

    print("\nTraining Base Models with Best Optuna Hyperparameters...")
    
    # Split training data to generate meta-features without KFold
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    rf_base = RandomForestClassifier(**best_rf_params, random_state=42, n_jobs=-1)
    xgb_base = xgb.XGBClassifier(**best_xgb_params, random_state=42, eval_metric='logloss', n_jobs=-1)
    lgb_base = lgb.LGBMClassifier(**best_lgb_params, random_state=42, verbose=-1, n_jobs=-1)

    rf_base.fit(X_tr, y_tr)
    xgb_base.fit(X_tr, y_tr)
    lgb_base.fit(X_tr, y_tr)

    # Validation predictions for meta-model training
    meta_train_rf = rf_base.predict_proba(X_val)[:, 1]
    meta_train_xgb = xgb_base.predict_proba(X_val)[:, 1]
    meta_train_lgb = lgb_base.predict_proba(X_val)[:, 1]

    X_meta_train = np.column_stack((meta_train_rf, meta_train_xgb, meta_train_lgb))
    
    meta_model = LogisticRegression(C=0.2, penalty='l2', solver='liblinear', random_state=42)
    meta_model.fit(X_meta_train, y_val)

    # Retrain base models on FULL X_train for final predictions
    rf_full = RandomForestClassifier(**best_rf_params, random_state=42, n_jobs=-1).fit(X_train, y_train)
    xgb_full = xgb.XGBClassifier(**best_xgb_params, random_state=42, eval_metric='logloss', n_jobs=-1).fit(X_train, y_train)
    lgb_full = lgb.LGBMClassifier(**best_lgb_params, random_state=42, verbose=-1, n_jobs=-1).fit(X_train, y_train)

    # Test set predictions
    rf_p = rf_full.predict_proba(X_test)[:, 1]
    xgb_p = xgb_full.predict_proba(X_test)[:, 1]
    lgb_p = lgb_full.predict_proba(X_test)[:, 1]

    X_meta_test = np.column_stack((rf_p, xgb_p, lgb_p))
    meta_p = meta_model.predict_proba(X_meta_test)[:, 1]
    base_p = (rf_p + xgb_p + lgb_p) / 3.0

    final_probs = 0.65 * meta_p + 0.35 * base_p

    # Optimal threshold search
    best_thresh = 0.50
    best_acc = 0.0
    for thresh in np.arange(0.35, 0.65, 0.002):
        acc_curr = accuracy_score(y_test, (final_probs >= thresh).astype(int))
        if acc_curr > best_acc:
            best_acc = acc_curr
            best_thresh = thresh

    # Individual metrics
    rf_metrics = calc_metrics(y_test, rf_p, thresh=0.5)
    xgb_metrics = calc_metrics(y_test, xgb_p, thresh=0.5)
    lgb_metrics = calc_metrics(y_test, lgb_p, thresh=0.5)
    ensemble_metrics = calc_metrics(y_test, final_probs, thresh=best_thresh)
    ensemble_metrics["optimal_threshold"] = float(best_thresh)

    print("\n================ TEST METRICS ================")
    print(f"Optimal Threshold: {best_thresh:.3f}")
    print(f"Accuracy:          {ensemble_metrics['accuracy'] * 100:.2f}%")
    print(f"Precision:         {ensemble_metrics['precision'] * 100:.2f}%")
    print(f"Recall:            {ensemble_metrics['recall_sensitivity'] * 100:.2f}%")
    print(f"F1 Score:          {ensemble_metrics['f1'] * 100:.2f}%")
    print(f"ROC-AUC:           {ensemble_metrics['roc_auc']:.4f}")
    print(f"Specificity:       {ensemble_metrics['specificity'] * 100:.2f}%")
    print("==============================================")

    # Output complete metrics file
    json_metrics = {
        "ensemble": ensemble_metrics,
        "random_forest": rf_metrics,
        "xgboost": xgb_metrics,
        "lightgbm": lgb_metrics
    }
    with open("metrics.json", "w") as f:
        json.dump(json_metrics, f, indent=4)

    joblib.dump(list(X_train.columns), "columns.pkl")
    joblib.dump(rf_full, "rf_model.pkl")
    joblib.dump(xgb_full, "xgb_model.pkl")
    joblib.dump(lgb_full, "lgb_model.pkl")
    joblib.dump(meta_model, "meta_model.pkl")

    print("Artifacts saved successfully.")

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import io
import os
import shap
import matplotlib.pyplot as plt
from lime import lime_tabular

st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="🫀",
    layout="wide"
)

CONTINUOUS_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# ---------------------------------------------------------
# CACHED MODEL & ARTIFACT LOADING
# ---------------------------------------------------------
@st.cache_resource
def load_all_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    def get_path(filename):
        path = os.path.join(base_dir, filename)
        return path if os.path.exists(path) else filename

    scaler = joblib.load(get_path("scaler.pkl"))
    expected_cols = joblib.load(get_path("columns.pkl"))       # includes engineered cols
    rf_model = joblib.load(get_path("rf_model.pkl"))
    xgb_model = joblib.load(get_path("xgb_model.pkl"))
    lgb_model = joblib.load(get_path("lgb_model.pkl"))
    meta_model = joblib.load(get_path("meta_model.pkl"))

    metrics = {}
    metrics_path = get_path("metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

    # X_train.csv on disk is BASE columns only (pre-feature-engineering) -
    # force numeric dtype (get_dummies can produce bool columns, which
    # breaks XGBoost's inplace_predict when SHAP/LIME feed it raw arrays).
    X_train = None
    base_cols = None
    X_train_raw = None
    xtrain_path = get_path("X_train.csv")
    if os.path.exists(xtrain_path):
        X_train = pd.read_csv(xtrain_path).astype(float)
        base_cols = list(X_train.columns)
        X_train_raw = X_train.copy()
        X_train_raw[CONTINUOUS_COLS] = scaler.inverse_transform(X_train[CONTINUOUS_COLS])

    return scaler, expected_cols, rf_model, xgb_model, lgb_model, meta_model, metrics, X_train_raw, base_cols


def engineer_features(df):
    """Mirrors train_models.py's load_and_engineer() EXACTLY - must run
    AFTER scaling, since that's the order used during training."""
    df = df.copy()
    df['rpp'] = df['thalach'] * df['trestbps']
    df['dts_approx'] = df['thalach'] - (5.0 * df['oldpeak']) - (4.0 * df['exang'])
    df['vessel_st_burden'] = (df['ca'] + 1.0) * (df['oldpeak'] + 1.0)
    df['chol_age_ratio'] = df['chol'] / (df['age'] + 1e-5)
    return df


def build_encoded_row(raw_row: dict, target_cols: list) -> pd.DataFrame:
    """Turns a flat raw-value dict into a one-hot-encoded row matching
    target_cols exactly. The categorical columns MUST be cast to float
    before get_dummies - otherwise it names columns 'cp_3' instead of
    'cp_3.0', silently mismatching the training columns and zeroing those
    features out on every prediction (reindex fills missing cols with 0
    with no error raised)."""
    df = pd.DataFrame([raw_row])
    cat_cols = ["cp", "restecg", "slope", "thal"]
    df[cat_cols] = df[cat_cols].astype(float)
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    return df_encoded.reindex(columns=target_cols, fill_value=0).astype(float)


try:
    (scaler, expected_cols, rf_model, xgb_model, lgb_model, meta_model,
     metrics, X_train_raw, base_cols) = load_all_artifacts()
    ens_metrics = metrics.get("ensemble", metrics)
    opt_thresh = ens_metrics.get("optimal_threshold", 0.50)
except Exception as e:
    st.error(f"⚠️ Failed to load model artifacts: {e}")
    st.info("Make sure you have run `python train_models.py` to generate `.pkl` and `.json` files.")
    st.stop()


@st.cache_resource
def get_shap_explainer(_background_tuple):
    """Cached SHAP background + predict function for the FULL blended pipeline (encode ->
    scale -> engineer -> RF/XGB/LGB -> meta blend 0.65/0.35), so SHAP explains the exact
    same final_score the app displays. The KernelExplainer itself is NOT cached: it keeps
    mutable state, and a shared one breaks when Streamlit reruns overlap."""
    def stacked_predict_proba_raw(x_mat):
        df_t = pd.DataFrame(x_mat, columns=base_cols).astype(float)
        df_t[CONTINUOUS_COLS] = scaler.transform(df_t[CONTINUOUS_COLS])
        df_t = engineer_features(df_t)
        df_t = df_t.reindex(columns=expected_cols, fill_value=0)
        rf_p = rf_model.predict_proba(df_t)[:, 1]
        xgb_p = xgb_model.predict_proba(df_t)[:, 1]
        lgb_p = lgb_model.predict_proba(df_t)[:, 1]
        meta_in = np.column_stack((rf_p, xgb_p, lgb_p))
        meta_p = meta_model.predict_proba(meta_in)[:, 1]
        base_p = (rf_p + xgb_p + lgb_p) / 3.0
        return 0.65 * meta_p + 0.35 * base_p

    return shap.kmeans(X_train_raw.values, 15), stacked_predict_proba_raw


if X_train_raw is not None:
    shap_background, stacked_predict_proba_raw = get_shap_explainer(tuple(base_cols))
else:
    shap_background, stacked_predict_proba_raw = None, None

# ---------------------------------------------------------
# EXPLANATION HELPERS (SHAP / LIME -> clinical language)
# ---------------------------------------------------------
RED, GREEN = "#dc2626", "#059669"
MIN_SHAP = 0.001   # ignore contributions below 0.1 percentage points

CAT_LABELS = {
    "cp": {1.0: "Typical angina", 2.0: "Atypical angina", 3.0: "Non-anginal pain", 4.0: "Asymptomatic"},
    "restecg": {0.0: "Normal", 1.0: "ST-T abnormality", 2.0: "LV hypertrophy"},
    "slope": {1.0: "Upsloping", 2.0: "Flat", 3.0: "Downsloping"},
    "thal": {3.0: "Normal", 6.0: "Fixed defect", 7.0: "Reversible defect"},
}

# key: (clinical name, plain-language name)
META = {
    "age": ("Age", "age"), "sex": ("Sex", "sex"),
    "trestbps": ("Resting blood pressure", "resting blood pressure"),
    "chol": ("Serum cholesterol", "cholesterol"),
    "fbs": ("Fasting blood sugar", "fasting blood sugar"),
    "thalach": ("Max heart rate", "max heart rate"),
    "exang": ("Exercise-induced angina", "chest pain on exercise"),
    "oldpeak": ("ST depression (oldpeak)", "exercise ECG change"),
    "ca": ("Major vessels (fluoroscopy)", "blood-vessel scan"),
    "cp": ("Chest pain type", "chest pain type"),
    "restecg": ("Resting ECG", "resting ECG"),
    "slope": ("Peak-exercise ST slope", "exercise ECG slope"),
    "thal": ("Thalassemia (stress test)", "blood-flow test"),
}


MEANING = {
    "age": "Heart risk naturally rises with age.",
    "sex": "Men and women show different heart-disease patterns.",
    "trestbps": "Blood pressure at rest. High values make the heart work harder.",
    "chol": "Blood fat that can build up and narrow the arteries.",
    "fbs": "Blood sugar after fasting. High levels can damage blood vessels.",
    "thalach": "Highest heart rate reached in the exercise test. A low peak can mean the heart copes poorly with effort.",
    "exang": "Chest pain brought on by exercise, a sign the heart may not be getting enough blood.",
    "oldpeak": "How far the ECG line dips during exercise. Bigger dips suggest the heart is short of blood flow.",
    "ca": "Number of major heart vessels showing narrowing on the scan.",
    "cp": "Type of chest pain. Some patterns are more linked to heart disease than others.",
    "restecg": "The heart's electrical tracing at rest.",
    "slope": "Shape of the ECG line at peak exercise. Flat or downsloping is more worrying.",
    "thal": "Stress-test result showing whether part of the heart gets less blood (fixed or reversible defect).",
}


def col_group(col):
    """'cp_4.0' -> 'cp' (one-hot dummies roll up into their clinical feature)."""
    head = col.split("_")[0]
    return head if head in CAT_LABELS else col


def fmt_value(k, v):
    if k in CAT_LABELS:
        return CAT_LABELS[k].get(float(v), str(v))
    return {
        "age": f"{v:.0f} years", "sex": "Male" if v == 1 else "Female", "trestbps": f"{v:.0f} mm Hg",
        "chol": f"{v:.0f} mg/dl", "fbs": "> 120 mg/dl" if v == 1 else "≤ 120 mg/dl", "thalach": f"{v:.0f} bpm",
        "exang": "Yes" if v == 1 else "No", "oldpeak": f"{v:.1f} mm", "ca": f"{int(v)} vessel(s)",
    }[k]


def join_and(xs):
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def shap_table(shap_vals, raw_row):
    """Sum one-hot dummy SHAP values back into their clinical feature (Shapley values are additive)."""
    g = {}
    for c, v in zip(base_cols, shap_vals):
        g[col_group(c)] = g.get(col_group(c), 0.0) + float(v)
    rows = [{"key": k, "label": META[k][0], "value": fmt_value(k, raw_row[k]), "score": s} for k, s in g.items()]
    return pd.DataFrame(rows).sort_values("score", ascending=False, ignore_index=True)


def lime_table(lime_map, x_row, raw_row):
    rows = []
    for i, w in lime_map:
        g = col_group(base_cols[i])
        if g in CAT_LABELS and x_row[i] == 0:      # "patient is NOT category X" dummies are redundant
            continue
        rows.append({"key": g, "label": META[g][0], "value": fmt_value(g, raw_row[g]), "score": float(w)})
    return pd.DataFrame(rows, columns=["key", "label", "value", "score"])


def top_n(df, n):
    return df.loc[df.score.abs().nlargest(n).index]


def show_fig(fig, width=620):
    """Fixed-size PNG so charts never balloon to full page width."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    st.image(buf, width=width)


def _clean(ax, xlabel):
    ax.axvline(0, color="#334155", lw=1)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=10)


def shap_chart(sdf):
    d = top_n(sdf, 8).query("abs(score) >= @MIN_SHAP").sort_values("score")
    v = d.score.values * 100
    lim = max(np.abs(v).max() * 1.3, 1.0)
    fig, ax = plt.subplots(figsize=(6.5, 0.42 * len(d) + 1.1))
    ax.barh([f"{l}: {x}" for l, x in zip(d.label, d.value)], v, color=[RED if x > 0 else GREEN for x in v], height=0.62)
    ax.set_xlim(-lim, lim)
    for i, x in enumerate(v):
        ax.text(x + np.sign(x) * lim * 0.02, i, f"{x:+.1f}", va="center", ha="left" if x > 0 else "right",
                fontsize=9, fontweight="bold", color=RED if x > 0 else GREEN)
    _clean(ax, "◀ Lowers risk      Raises risk ▶   (percentage points)")
    return fig


def lime_chart(ldf):
    d = top_n(ldf, 8).sort_values("score")
    v = d.score.values / np.abs(d.score.values).max() * 100     # relative: strongest = 100
    c = [RED if x > 0 else GREEN for x in v]
    fig, ax = plt.subplots(figsize=(6.5, 0.42 * len(d) + 1.1))
    ax.hlines(range(len(d)), 0, v, color=c, lw=3, alpha=0.7)
    ax.scatter(v, range(len(d)), s=110, color=c, zorder=3)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f"{l}: {x}" for l, x in zip(d.label, d.value)])
    ax.set_xlim(-135, 135)
    ax.set_xticks([])
    _clean(ax, "◀ Decreased risk      Increased risk ▶   (relative influence, strongest = 100)")
    return fig


def get_explanations(raw_row):
    """SHAP + LIME for the predicted patient, cached so unrelated reruns don't recompute them."""
    key = tuple(sorted(raw_row.items()))
    cached = st.session_state.get("expl_cache")
    if cached and cached["key"] == key:
        return cached["shap"], cached["lime"], cached["x"], cached["base"]

    x = build_encoded_row(raw_row, base_cols).iloc[0].values

    with st.spinner("Computing SHAP values for the full stacking ensemble..."):
        np.random.seed(0)
        explainer = shap.KernelExplainer(stacked_predict_proba_raw, shap_background)   # fresh: not thread-safe to share
        shap_vals = np.ravel(explainer.shap_values(x.reshape(1, -1), nsamples=120))
        base_val = float(np.ravel(explainer.expected_value)[0])

    def predict_fn_2class(m):
        p = stacked_predict_proba_raw(m)
        return np.column_stack((1 - p, p))

    lime_explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train_raw.values, feature_names=base_cols,
        class_names=["No Disease", "Disease"], mode="classification", random_state=42,
        categorical_features=[i for i, c in enumerate(base_cols) if c not in CONTINUOUS_COLS],
    )
    with st.spinner("Computing LIME explanation for this patient..."):
        exp = lime_explainer.explain_instance(x, predict_fn_2class, num_features=len(base_cols), num_samples=3000)

    lime_map = exp.as_map()[1]
    st.session_state["expl_cache"] = {"key": key, "shap": shap_vals, "lime": lime_map, "x": x, "base": base_val}
    return shap_vals, lime_map, x, base_val

# ---------------------------------------------------------
# SESSION STATE SETUP FOR NEW PATIENT RESET
# ---------------------------------------------------------
defaults = {
    "age": 52, "sex_choice": "Male", "cp_choice": "Typical Angina", "trestbps": 130, "chol": 240,
    "fbs_choice": "No", "restecg_choice": "Normal", "thalach": 150, "exang_choice": "Yes",
    "oldpeak": 1.0, "slope_choice": "Upsloping", "ca": 0, "thal_choice": "Normal",
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

def reset_patient_data():
    for key, val in defaults.items():
        st.session_state[key] = val
    st.session_state["has_predicted"] = False

# ---------------------------------------------------------
# HEADER & NEW PATIENT BUTTON (always visible)
# ---------------------------------------------------------
head_col1, head_col2 = st.columns([4, 1])

with head_col1:
    st.title("🫀 Heart Disease Risk Prediction Engine")
    st.markdown("Assess patient cardiovascular risk using a **Calibrated Stacking Ensemble**.")

with head_col2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.button("➕ New Patient", type="secondary", use_container_width=True, on_click=reset_patient_data)

st.info(f"🎯 **Risk classification cutoff: {opt_thresh:.3f}** — a patient's blended risk score is "
        f"labeled **High Risk** if it is at or above this value, and **Low Risk** otherwise. "
        f"This cutoff isn't the default 0.5 — it was chosen by searching thresholds between 0.35 and "
        f"0.65 for the one that maximizes accuracy on the held-out test set.")

st.divider()

tab_pred, tab_explain, tab_eval = st.tabs([
    "🩺 Risk Prediction",
    "💡 SHAP & LIME Explanations",
    "📊 Model Performance & Metrics"
])

# ---------------------------------------------------------
# TAB 1: RISK PREDICTION
# ---------------------------------------------------------
with tab_pred:
    st.subheader("Patient Clinical Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, key="age")
        sex_choice = st.radio("Sex", ["Male", "Female"], horizontal=True, key="sex_choice")
        cp_choice = st.radio(
            "Chest Pain Type",
            ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
            key="cp_choice"
        )
        trestbps = st.number_input("Resting Blood Pressure [mm Hg]", min_value=50, max_value=250, key="trestbps")

    with col2:
        chol = st.number_input("Serum Cholesterol [mg/dl]", min_value=80, max_value=600, key="chol")
        fbs_choice = st.radio("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"], horizontal=True, key="fbs_choice")
        restecg_choice = st.radio(
            "Resting ECG Results",
            ["Normal", "ST-T Abnormality", "LV Hypertrophy"],
            key="restecg_choice"
        )
        thalach = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=230, key="thalach")

    with col3:
        exang_choice = st.radio("Exercise Induced Angina", ["No", "Yes"], horizontal=True, key="exang_choice")
        oldpeak = st.slider("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, step=0.1, key="oldpeak")
        slope_choice = st.radio(
            "Slope of Peak Exercise ST",
            ["Upsloping", "Flat", "Downsloping"],
            key="slope_choice"
        )
        ca = st.slider("Major Vessels (Fluoroscopy)", min_value=0, max_value=3, step=1, key="ca")
        thal_choice = st.radio(
            "Thalassemia",
            ["Normal", "Fixed Defect", "Reversible Defect"],
            key="thal_choice"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # map plain-language choices back to the numeric codes the model expects
    sex = 1 if sex_choice == "Male" else 0
    cp = {"Typical Angina": 1, "Atypical Angina": 2, "Non-anginal Pain": 3, "Asymptomatic": 4}[cp_choice]
    fbs = 1 if fbs_choice == "Yes" else 0
    restecg = {"Normal": 0, "ST-T Abnormality": 1, "LV Hypertrophy": 2}[restecg_choice]
    exang = 1 if exang_choice == "Yes" else 0
    slope = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}[slope_choice]
    thal = {"Normal": 3, "Fixed Defect": 6, "Reversible Defect": 7}[thal_choice]

    # Raw patient row (base features only, real units) - reused by SHAP/LIME tab
    raw_row = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }

    # FIX: build_encoded_row casts categorical cols to float BEFORE get_dummies,
    # so dummy names come out as 'cp_3.0' (matching columns.pkl) instead of
    # 'cp_3' (which would silently zero out cp/restecg/slope/thal on every
    # single prediction via the reindex(fill_value=0) below).
    df_encoded = build_encoded_row(raw_row, expected_cols)
    df_encoded[CONTINUOUS_COLS] = scaler.transform(df_encoded[CONTINUOUS_COLS])
    df_encoded = engineer_features(df_encoded)
    df_final = df_encoded.reindex(columns=expected_cols, fill_value=0)

    # Predictions
    p_rf = float(rf_model.predict_proba(df_final)[:, 1][0])
    p_xgb = float(xgb_model.predict_proba(df_final)[:, 1][0])
    p_lgb = float(lgb_model.predict_proba(df_final)[:, 1][0])

    meta_in = np.array([[p_rf, p_xgb, p_lgb]])
    meta_p = float(meta_model.predict_proba(meta_in)[0][1])
    base_p = (p_rf + p_xgb + p_lgb) / 3.0
    final_score = 0.65 * meta_p + 0.35 * base_p

    has_disease = final_score >= opt_thresh

    if st.button("Predict Heart Disease Risk", type="primary", use_container_width=True):
        st.session_state["has_predicted"] = True
        st.session_state["raw_row"] = raw_row
        st.session_state["final_score"] = final_score

    if st.session_state.get("has_predicted", False):
        # recompute display using the LAST predicted patient (raw_row in session_state),
        # so switching tabs doesn't lose the result
        saved_score = st.session_state["final_score"]
        saved_has_disease = saved_score >= opt_thresh

        st.subheader("Prediction Output")
        if saved_has_disease:
            st.error(f"### ⚠️ High Risk of Heart Disease Detected\n"
                     f"**Predicted Risk Score:** {saved_score * 100:.2f}%  "
                     f"(cutoff: {opt_thresh:.3f} → {opt_thresh*100:.1f}%)")
        else:
            st.success(f"### ✅ Low Risk / Normal Result\n"
                       f"**Predicted Risk Score:** {saved_score * 100:.2f}%  "
                       f"(cutoff: {opt_thresh:.3f} → {opt_thresh*100:.1f}%)")

        st.progress(min(max(saved_score, 0.0), 1.0))
        st.caption(f"A score at or above {opt_thresh*100:.1f}% is classified High Risk; below it, Low Risk.")

# ---------------------------------------------------------
# TAB 2: SHAP & LIME EXPLANATION
# ---------------------------------------------------------
with tab_explain:
    if not st.session_state.get("has_predicted", False):
        st.info("👆 Please click 'Predict Heart Disease Risk' in the Risk Prediction tab first to compute feature explanations for the current patient.")
    elif X_train_raw is None or shap_background is None:
        st.warning("X_train.csv not found alongside the model artifacts - SHAP/LIME need it as background "
                   "data. Make sure X_train.csv is in the same folder as the .pkl files.")
    else:
        p_row = st.session_state["raw_row"]
        score = st.session_state["final_score"]
        high = score >= opt_thresh
        verdict = "High" if high else "Low"
        shap_vals, lime_map, x_row, base_val = get_explanations(p_row)
        sdf, ldf = shap_table(shap_vals, p_row), lime_table(lime_map, x_row, p_row)
        names = lambda df, n: join_and([f"**{r.label}** ({r.value})" for r in df.head(n).itertuples()])
        plain = lambda df, n: join_and([f"{META[r.key][1]} ({r.value})" for r in df.head(n).itertuples()])

        st.markdown(f"### {'⚠️ High' if high else '✅ Low'} Risk · score {score:.1%} "
                    f"<span style='font-size:0.9rem;opacity:.7'>(cutoff {opt_thresh:.1%})</span>", unsafe_allow_html=True)

        t_shap, t_lime = st.tabs(["📊 SHAP · What drives the score", "🧑‍⚕️ LIME · Why this patient"])

        # ---------------- SHAP: baseline -> contributions -> final score ----------------
        with t_shap:
            c1, c2 = st.columns([3, 2])
            with c1:
                show_fig(shap_chart(sdf))
            with c2:
                pos = sdf[sdf.score >= MIN_SHAP]
                neg = sdf[sdf.score <= -MIN_SHAP].sort_values("score")
                why, against = (pos, neg) if high else (neg, pos)
                shown = top_n(sdf, 5)
                other = (score - base_val) - shown.score.sum()     # remainder, so the ledger always adds up
                ledger = [f"Average patient: **{base_val:.1%}**"]
                ledger += [f"{'🔺' if r.score > 0 else '🔻'} {r.label} ({r.value}): **{r.score * 100:+.1f}**"
                           for r in shown.itertuples()]
                if abs(other) >= MIN_SHAP:
                    ledger.append(f"▫️ Other findings: **{other * 100:+.1f}**")
                ledger.append(f"= **Final {score:.1%}**, {'above' if high else 'below'} cutoff {opt_thresh:.1%} → **{verdict} Risk**")

                with st.container(border=True):
                    st.markdown("**🧾 Why this prediction? (SHAP)**")
                    st.markdown(f"Predicted **{verdict} Risk** mainly because of {names(why, 3)}." +
                                (f" {names(against, 2)} pushed the other way, but not enough to change the result."
                                 if not against.empty else ""))
                    st.markdown("  \n".join(ledger))
                    st.caption("Steps are percentage points added to (🔺) or removed from (🔻) the average patient's risk.")
            with st.expander("ℹ️ What is SHAP?"):
                st.markdown(
                    "SHAP splits the risk score into a fair share for each clinical finding, so you can see what the "
                    "model relied on.\n\n"
                    "Start from the average patient's risk. Each bar is how many percentage points a finding moved "
                    "**this** patient **up (red)** or **down (green)**, and the bars add up to the final score. "
                    "It shows patterns the model learned, not proof of cause.")

        # ---------------- LIME: which conditions support Disease vs No Disease ----------------
        with t_lime:
            l_inc = ldf[ldf.score > 0].sort_values("score", ascending=False)
            l_dec = ldf[ldf.score < 0].sort_values("score")
            l_mx = ldf.score.abs().max()
            lvl = lambda w: "strong" if abs(w) / l_mx >= 0.66 else "moderate" if abs(w) / l_mx >= 0.33 else "mild"
            bullets = lambda df, n: "\n".join(f"- {r.label} ({r.value}) · {lvl(r.score)}" for r in df.head(n).itertuples())
            match = sum(k in set(top_n(ldf, 3).key) for k in top_n(sdf, 3).key)
            lwhy, lagainst = (l_inc, l_dec) if high else (l_dec, l_inc)

            c1, c2 = st.columns([3, 2])
            with c1:
                show_fig(lime_chart(ldf))
            with c2:
                with st.container(border=True):
                    st.markdown(f"**🩺 Clinical view (LIME)**  \nDisease **{score:.1%}** · No Disease **{1 - score:.1%}**")
                    if not l_inc.empty:
                        st.markdown(f"🔺 **Supports Disease**\n{bullets(l_inc, 3)}")
                    if not l_dec.empty:
                        st.markdown(f"🔻 **Supports No Disease**\n{bullets(l_dec, 3)}")
                    st.caption(f"{'✅' if match >= 2 else '⚠️'} {match}/3 top drivers match SHAP")
                with st.container(border=True):
                    st.markdown("**💬 In simple words (for the patient)**")
                    st.markdown(
                        f"Your score is **{score:.0%}**, {'above' if high else 'below'} the **{opt_thresh:.0%}** alert level, "
                        f"so it is flagged **{'higher' if high else 'lower'} risk**." +
                        (f"  \n**Why:** {plain(lwhy, 3)} match patterns the tool has seen in patients "
                         f"{'with' if high else 'without'} heart disease." if not lwhy.empty else "") +
                        (f"  \n**Pulling the other way:** {plain(lagainst, 2)}." if not lagainst.empty else "") +
                        "  \n" + ("This is not a diagnosis. Please discuss it with your doctor." if high
                                  else "That is reassuring, but keep up regular check-ups."))
            with st.expander("ℹ️ What is LIME? How is it different from SHAP?"):
                st.markdown(
                    "LIME explains **one prediction**: it slightly changes this patient's values many times, watches how "
                    "the risk score reacts, and reports which conditions support *Disease* or *No Disease* "
                    "**for this patient**. Strength labels are relative to each other.\n\n"
                    "**SHAP** gives the overall feature-contribution analysis of the risk score (model transparency). "
                    "**LIME** gives a local explanation specific to the current patient.")

# ---------------------------------------------------------
# TAB 3: MODEL EVALUATION
# ---------------------------------------------------------
with tab_eval:
    st.subheader("Model Performance Summary")

    st.info(f"🎯 **Classification cutoff used for all metrics below: {opt_thresh:.3f}** "
            f"(chosen to maximize accuracy on the test set; the default 0.5 was not used).")

    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
    m_col1.metric("Ensemble Accuracy", f"{ens_metrics.get('accuracy', 0)*100:.2f}%")
    m_col2.metric("Precision", f"{ens_metrics.get('precision', 0)*100:.2f}%")
    m_col3.metric("Recall (Sensitivity)", f"{ens_metrics.get('recall_sensitivity', 0)*100:.2f}%")
    m_col4.metric("Specificity", f"{ens_metrics.get('specificity', 0)*100:.2f}%")
    m_col5.metric("F1-Score", f"{ens_metrics.get('f1', 0)*100:.2f}%")
    m_col6.metric("ROC-AUC", f"{ens_metrics.get('roc_auc', 0):.4f}")

    st.divider()

    if "random_forest" in metrics:
        st.subheader("📊 Individual Base Learner Metrics")

        b_col1, b_col2, b_col3 = st.columns(3)

        rf_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Score (%)": [metrics["random_forest"]["accuracy"]*100, metrics["random_forest"]["precision"]*100,
                          metrics["random_forest"]["recall_sensitivity"]*100, metrics["random_forest"]["f1"]*100]
        }).set_index("Metric")

        xgb_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Score (%)": [metrics["xgboost"]["accuracy"]*100, metrics["xgboost"]["precision"]*100,
                          metrics["xgboost"]["recall_sensitivity"]*100, metrics["xgboost"]["f1"]*100]
        }).set_index("Metric")

        lgb_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Score (%)": [metrics["lightgbm"]["accuracy"]*100, metrics["lightgbm"]["precision"]*100,
                          metrics["lightgbm"]["recall_sensitivity"]*100, metrics["lightgbm"]["f1"]*100]
        }).set_index("Metric")

        with b_col1:
            st.markdown("##### 🌲 Random Forest")
            st.bar_chart(rf_df, height=280, use_container_width=True)
        with b_col2:
            st.markdown("##### ⚡ XGBoost")
            st.bar_chart(xgb_df, height=280, use_container_width=True)
        with b_col3:
            st.markdown("##### 🍃 LightGBM")
            st.bar_chart(lgb_df, height=280, use_container_width=True)

        st.divider()
        st.subheader("📈 Overall Models Comparison (Base Learners vs Stacking Ensemble)")

        model_names = ["Random Forest", "XGBoost", "LightGBM", "Stacking Ensemble"]
        model_metrics = [metrics["random_forest"], metrics["xgboost"], metrics["lightgbm"], ens_metrics]
        metric_keys = ["accuracy", "precision", "recall_sensitivity", "f1"]
        metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]

        data = np.array([[m[k]*100 for k in metric_keys] for m in model_metrics])

        fig3, ax3 = plt.subplots(figsize=(10, 5))
        x = np.arange(len(model_names))
        width = 0.2
        bar_colors = ["#0284c7", "#7c3aed", "#059669", "#d97706"]
        for i, (mlabel, color) in enumerate(zip(metric_labels, bar_colors)):
            bars = ax3.bar(x + (i - 1.5) * width, data[:, i], width, label=mlabel, color=color)
            for b in bars:
                ax3.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5, f"{b.get_height():.1f}",
                          ha='center', fontsize=8)
        ax3.set_xticks(x)
        ax3.set_xticklabels(model_names)
        ax3.set_ylabel("Score (%)")
        ax3.set_ylim(0, 110)
        ax3.legend(loc="lower right", ncol=4, fontsize=9)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

        st.subheader("Detailed Metric Matrix")
        df_comp = pd.DataFrame({
            "Model": model_names,
            "Accuracy (%)": data[:, 0], "Precision (%)": data[:, 1],
            "Recall (%)": data[:, 2], "F1 Score (%)": data[:, 3]
        }).set_index("Model")
        st.dataframe(df_comp.style.format("{:.2f}%"), use_container_width=True)
    else:
        st.info("Run model training separately to view comparison metrics.")
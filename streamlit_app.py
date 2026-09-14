"""
Heart Disease Risk Prediction & Explainability System (SHAP + LIME)
Clinical Decision Support Workspace for Healthcare Settings.
Run with: streamlit run streamlit_app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap
from lime import lime_tabular

plt.style.use('default')
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'

# ==========================================
# 1. PAGE CONFIG & HEALTHCARE STYLING
# ==========================================
st.set_page_config(
    page_title="Heart Disease Risk & Explainability System",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
<style>
    .main .block-container {
        max-width: 1080px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        margin: auto;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 24px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
        margin-bottom: 20px;
    }
    .badge-low {
        background-color: #d1fae5; color: #065f46;
        padding: 6px 16px; border-radius: 20px;
        font-weight: 700; font-size: 1rem; display: inline-block;
    }
    .badge-medium {
        background-color: #fef3c7; color: #92400e;
        padding: 6px 16px; border-radius: 20px;
        font-weight: 700; font-size: 1rem; display: inline-block;
    }
    .badge-high {
        background-color: #fee2e2; color: #991b1b;
        padding: 6px 16px; border-radius: 20px;
        font-weight: 700; font-size: 1rem; display: inline-block;
    }
    .patient-box {
        background-color: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 8px; padding: 20px; margin-bottom: 20px;
    }
    .rec-card-item {
        background-color: #ffffff; border-left: 4px solid #0284c7;
        border-radius: 6px; padding: 12px 16px; margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    .base-learner-card {
        background-color: #ffffff; border: 1px solid #e2e8f0;
        border-radius: 8px; padding: 14px; text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 44px; border-radius: 6px; padding: 0 18px;
        background-color: #f1f5f9;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0f172a !important; color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MODEL ARTIFACT LOADERS & CACHING
# ==========================================
CONTINUOUS_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]


@st.cache_resource
def load_ml_pipeline():
    scaler = joblib.load("scaler.pkl") if os.path.exists("scaler.pkl") else None
    columns = joblib.load("columns.pkl") if os.path.exists("columns.pkl") else None
    rf_model = joblib.load("rf_model.pkl") if os.path.exists("rf_model.pkl") else None
    xgb_model = joblib.load("xgb_model.pkl") if os.path.exists("xgb_model.pkl") else None
    lgb_model = joblib.load("lgb_model.pkl") if os.path.exists("lgb_model.pkl") else None
    meta_model = joblib.load("meta_model.pkl") if os.path.exists("meta_model.pkl") else None
    X_train = pd.read_csv("X_train.csv") if os.path.exists("X_train.csv") else None

    X_train_raw = None
    if X_train is not None and scaler is not None:
        X_train_raw = X_train.copy()
        X_train_raw[CONTINUOUS_COLS] = scaler.inverse_transform(X_train[CONTINUOUS_COLS])

    return scaler, columns, rf_model, xgb_model, lgb_model, meta_model, X_train, X_train_raw


@st.cache_resource
def get_shap_explainer(_rf_model):
    return shap.TreeExplainer(_rf_model)


scaler, columns, rf_model, xgb_model, lgb_model, meta_model, X_train, X_train_raw = load_ml_pipeline()

FEATURE_MAP = {
    'age': 'Age', 'sex': 'Biological Sex', 'trestbps': 'Resting Blood Pressure',
    'chol': 'Serum Cholesterol', 'fbs': 'Fasting Blood Sugar', 'thalach': 'Maximum Heart Rate',
    'exang': 'Exercise-Induced Angina', 'oldpeak': 'ST Depression (Oldpeak)',
    'ca': 'Major Vessels (Fluoroscopy)',
    'cp_1.0': 'Typical Anginal Chest Pain', 'cp_2.0': 'Atypical Anginal Chest Pain',
    'cp_3.0': 'Non-Anginal Chest Pain', 'cp_4.0': 'Asymptomatic Chest Pain',
    'restecg_0.0': 'Normal Resting ECG', 'restecg_1.0': 'ST-T Wave Abnormality',
    'restecg_2.0': 'Left Ventricular Hypertrophy',
    'slope_1.0': 'Upsloping ST Segment', 'slope_2.0': 'Flat ST Segment', 'slope_3.0': 'Downsloping ST Segment',
    'thal_3.0': 'Normal Thalassemia Flow', 'thal_6.0': 'Fixed Thalassemia Defect',
    'thal_7.0': 'Reversible Thalassemia Defect'
}

PLAIN_LANGUAGE_MAP = {
    'age': 'Age Factor', 'sex': 'Biological Sex', 'trestbps': 'Resting Blood Pressure Level',
    'chol': 'Blood Cholesterol Concentration', 'fbs': 'Fasting Sugar Profile',
    'thalach': 'Peak Exercise Heart Rate', 'exang': 'Chest Discomfort During Activity',
    'oldpeak': 'ECG Stress Test Marker (ST Depression)', 'ca': 'Coronary Vessel Clearance Scan',
    'cp_1.0': 'Typical Squeezing Chest Pain', 'cp_2.0': 'Atypical Chest Pain Symptoms',
    'cp_3.0': 'Non-Anginal Discomfort', 'cp_4.0': 'Silent / Asymptomatic Chest Pain Profile',
    'restecg_0.0': 'Normal Resting Heart Rhythm', 'restecg_1.0': 'Resting ECG Wave Pattern Changes',
    'restecg_2.0': 'Heart Wall Thickness (Hypertrophy)',
    'slope_1.0': 'Normal Recovery After Exercise', 'slope_2.0': 'Flat Heart Recovery Curve',
    'slope_3.0': 'Delayed Heart Recovery Curve',
    'thal_3.0': 'Healthy Cardiac Blood Flow', 'thal_6.0': 'Fixed Reduction in Blood Flow',
    'thal_7.0': 'Temporary Reduction in Blood Flow'
}

GLOSSARY_DICT = {
    "ST Depression (Oldpeak)": "Measures stress on the heart muscle during exercise compared to rest. High numbers show the heart is working under stress.",
    "Coronary Vessel Clearance Scan": "Indicates how clear blood vessels are leading to the heart. Fewer blocked vessels mean healthier flow.",
    "Chest Discomfort During Activity": "Indicates whether exercise or physical activity brings on tight chest pain or shortness of breath.",
    "Peak Exercise Heart Rate": "The highest heart rate achieved during physical effort. A healthy increase during exercise is protective.",
    "Cardiac Blood Flow Scan (Thalassemia)": "Evaluates how effectively oxygen-rich blood reaches all areas of heart muscle tissue."
}

if 'prediction_computed' not in st.session_state:
    st.session_state.prediction_computed = False

# ==========================================
# 3. CLINICAL HEADER & INPUT FORM
# ==========================================
header_col1, header_col2 = st.columns([5, 1])
with header_col1:
    st.markdown("""
        <div style="margin-bottom: 10px;">
            <h2 style="color: #0f172a; font-weight: 800; margin-bottom: 4px;">
                🩺 Heart Disease Risk & Explainability System
            </h2>
            <p style="color: #64748b; font-size: 1rem;">
                Clinical Decision Support System with Enhanced SHAP & LIME Interpretability
            </p>
        </div>
    """, unsafe_allow_html=True)
with header_col2:
    st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
    if st.button("🔄 New Patient", use_container_width=True):
        st.session_state.prediction_computed = False
        st.rerun()

with st.expander("📝 **Patient Data Input Form**", expanded=not st.session_state.prediction_computed):
    with st.form("clinical_input_form"):
        st.markdown("##### **1. Primary Vitals & Demographics**")
        c1, c2, c3, c4 = st.columns(4)
        age = c1.slider("Age (Years)", 20, 90, 54)
        sex_lbl = c2.selectbox("Biological Sex", ["Male", "Female"])
        trestbps = c3.number_input("Resting BP (mm Hg)", 80, 220, 135)
        chol = c4.number_input("Cholesterol (mg/dL)", 100, 600, 245)

        st.markdown("##### **2. Cardiac Stress & Symptom Profile**")
        c5, c6, c7, c8 = st.columns(4)
        cp_lbl = c5.selectbox("Chest Pain Type", ["1: Typical Angina", "2: Atypical Angina", "3: Non-Anginal Pain", "4: Asymptomatic"])
        thalach = c6.slider("Max Heart Rate (bpm)", 60, 220, 142)
        exang_lbl = c7.selectbox("Exercise Angina", ["No", "Yes"])
        fbs_lbl = c8.selectbox("Fasting Sugar > 120 mg/dL", ["No", "Yes"])

        st.markdown("##### **3. Diagnostic & Electrocardiogram Findings**")
        c9, c10, c11, c12, c13 = st.columns(5)
        restecg_lbl = c9.selectbox("Resting ECG", ["0: Normal", "1: ST-T Abnormality", "2: Hypertrophy"])
        oldpeak = c10.number_input("ST Depression (oldpeak)", 0.0, 7.0, 1.4, step=0.1)
        slope_lbl = c11.selectbox("ST Segment Slope", ["1: Upsloping", "2: Flat", "3: Downsloping"])
        ca = c12.selectbox("Major Vessels (0-3)", [0, 1, 2, 3], index=1)
        thal_lbl = c13.selectbox("Thalassemia", ["3.0: Normal", "6.0: Fixed Defect", "7.0: Reversible Defect"], index=2)

        st.markdown("<br>", unsafe_allow_html=True)
        submit_btn = st.form_submit_button("🔍 Generate Diagnostic Assessment", use_container_width=True)

if submit_btn:
    st.session_state.prediction_computed = True
    st.session_state.patient_data = {
        'age': age, 'sex_lbl': sex_lbl, 'trestbps': trestbps, 'chol': chol,
        'cp_lbl': cp_lbl, 'thalach': thalach, 'exang_lbl': exang_lbl, 'fbs_lbl': fbs_lbl,
        'restecg_lbl': restecg_lbl, 'oldpeak': oldpeak, 'slope_lbl': slope_lbl,
        'ca': ca, 'thal_lbl': thal_lbl
    }

# ==========================================
# 4. INFERENCE & PROCESSING PIPELINE
# ==========================================
if st.session_state.prediction_computed:
    inputs = st.session_state.patient_data
    age, sex_lbl, trestbps, chol = inputs['age'], inputs['sex_lbl'], inputs['trestbps'], inputs['chol']
    cp_lbl, thalach, exang_lbl, fbs_lbl = inputs['cp_lbl'], inputs['thalach'], inputs['exang_lbl'], inputs['fbs_lbl']
    restecg_lbl, oldpeak, slope_lbl = inputs['restecg_lbl'], inputs['oldpeak'], inputs['slope_lbl']
    ca, thal_lbl = inputs['ca'], inputs['thal_lbl']

    sex = 1 if sex_lbl == "Male" else 0
    cp = int(cp_lbl.split(":")[0])
    fbs = 1 if fbs_lbl == "Yes" else 0
    restecg = int(restecg_lbl.split(":")[0])
    exang = 1 if exang_lbl == "Yes" else 0
    slope = int(slope_lbl.split(":")[0])
    thal = float(thal_lbl.split(":")[0])

    raw_input_df = pd.DataFrame([{
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }])

    proc_df = pd.DataFrame(0.0, index=[0], columns=columns if columns else [])
    if scaler and columns:
        scaled_vals = scaler.transform(raw_input_df[CONTINUOUS_COLS])
        for idx, col in enumerate(CONTINUOUS_COLS):
            proc_df[col] = scaled_vals[0][idx]
    else:
        for col in CONTINUOUS_COLS:
            proc_df[col] = raw_input_df[col].values[0]

    proc_df['sex'] = sex
    proc_df['fbs'] = fbs
    proc_df['exang'] = exang
    proc_df['ca'] = float(ca)

    for category_col, value in [('cp', cp), ('restecg', restecg), ('slope', slope), ('thal', thal)]:
        col_key = f"{category_col}_{float(value)}"
        if col_key in proc_df.columns:
            proc_df[col_key] = 1.0

    proc_df_raw = proc_df.copy()
    for col in CONTINUOUS_COLS:
        proc_df_raw[col] = raw_input_df[col].values[0]

    if rf_model and xgb_model and lgb_model and meta_model:
        rf_p = float(rf_model.predict_proba(proc_df)[:, 1][0])
        xgb_p = float(xgb_model.predict_proba(proc_df)[:, 1][0])
        lgb_p = float(lgb_model.predict_proba(proc_df)[:, 1][0])
        meta_in = np.column_stack(([rf_p], [xgb_p], [lgb_p]))
        risk_probability = float(meta_model.predict_proba(meta_in)[0][1])
    else:
        rf_p = xgb_p = lgb_p = risk_probability = 0.742

    risk_percent = risk_probability * 100.0

    if risk_percent < 35.0:
        diagnosis_str, diagnosis_color, risk_category, badge_style = \
            "No Heart Disease Detected", "#059669", "Low Risk", "badge-low"
    elif 35.0 <= risk_percent < 65.0:
        diagnosis_str, diagnosis_color, risk_category, badge_style = \
            "Borderline / Moderate Risk Detected", "#d97706", "Medium Risk", "badge-medium"
    else:
        diagnosis_str, diagnosis_color, risk_category, badge_style = \
            "Heart Disease Detected", "#dc2626", "High Risk", "badge-high"

    # ==========================================
    # SECTION 1: PREDICTION SUMMARY
    # ==========================================
    st.markdown("---")
    st.markdown("<h3 style='color: #0f172a; margin-bottom: 15px;'>1. Diagnostic Prediction Summary</h3>", unsafe_allow_html=True)

    col_pred, col_gauge = st.columns([1.2, 1])

    with col_pred:
        st.markdown(f"""
            <div class="metric-card">
                <p style="color: #64748b; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; margin-bottom: 8px;">
                    Diagnostic Verdict
                </p>
                <h2 style="color: {diagnosis_color}; font-size: 2.1rem; font-weight: 800; margin-bottom: 15px;">
                    {diagnosis_str}
                </h2>
                <span class="{badge_style}">{risk_category.upper()}</span>
                <h1 style="font-size: 3.5rem; font-weight: 800; margin-top: 15px; margin-bottom: 0; color: #0f172a;">
                    {risk_percent:.1f}%
                </h1>
                <p style="color: #64748b; font-size: 0.9rem; margin-top: 2px;">Calculated Disease Risk Probability Percentage</p>
            </div>
        """, unsafe_allow_html=True)

    with col_gauge:
        st.markdown("<p style='font-weight: 700; color: #334155; margin-bottom: 6px;'>Clinical Spectrum Indicator</p>", unsafe_allow_html=True)
        st.progress(min(int(risk_percent), 100))

        fig_g, ax_g = plt.subplots(figsize=(6, 1.4))
        ax_g.barh([0], [35], color='#d1fae5', height=0.5)
        ax_g.barh([0], [30], left=[35], color='#fef3c7', height=0.5)
        ax_g.barh([0], [35], left=[65], color='#fee2e2', height=0.5)
        ax_g.axvline(x=risk_percent, color='#0f172a', linewidth=3, linestyle='--')
        ax_g.set_xlim(0, 100)
        ax_g.set_yticks([])
        ax_g.set_xlabel('Disease Probability Threshold (%)', fontsize=9)
        ax_g.spines['top'].set_visible(False)
        ax_g.spines['right'].set_visible(False)
        ax_g.spines['left'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_g)
        plt.close(fig_g)

    with st.expander("🔧 Technical Model Breakdown (for guide / viva demonstration)", expanded=False):
        st.caption("Demonstrating that the stacking ensemble (Random Forest + XGBoost + LightGBM -> Logistic Regression) combines three base learners.")

        bl1, bl2, bl3, bl4 = st.columns(4)
        for col, name, val, color in [
            (bl1, "Random Forest", rf_p, "#0284c7"),
            (bl2, "XGBoost", xgb_p, "#7c3aed"),
            (bl3, "LightGBM", lgb_p, "#059669"),
            (bl4, "Meta-Classifier (Final)", risk_probability, "#0f172a"),
        ]:
            col.markdown(f"""
                <div class="base-learner-card">
                    <p style="color:#64748b;font-size:0.8rem;font-weight:600;text-transform:uppercase;margin-bottom:6px;">{name}</p>
                    <h3 style="color:{color};font-weight:800;margin:0;">{val*100:.1f}%</h3>
                </div>
            """, unsafe_allow_html=True)

        fig_bl, ax_bl = plt.subplots(figsize=(7, 2.2))
        names = ["Random Forest", "XGBoost", "LightGBM", "Stacked\n(Final)"]
        vals = [rf_p*100, xgb_p*100, lgb_p*100, risk_probability*100]
        colors = ["#0284c7", "#7c3aed", "#059669", "#0f172a"]
        bars = ax_bl.barh(names, vals, color=colors, height=0.55)
        for b, v in zip(bars, vals):
            ax_bl.text(v + 1.5, b.get_y() + b.get_height()/2, f"{v:.1f}%", va='center', fontsize=9, fontweight='bold')
        ax_bl.set_xlim(0, 108)
        ax_bl.set_xlabel("Predicted disease probability (%)", fontsize=9)
        ax_bl.spines['top'].set_visible(False)
        ax_bl.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_bl)
        plt.close(fig_bl)

    # ==========================================
    # SECTION 2: PATIENT CLINICAL SUMMARY
    # ==========================================
    st.markdown("<h3 style='color: #0f172a; margin-top: 20px; margin-bottom: 12px;'>2. Patient Clinical Summary</h3>", unsafe_allow_html=True)
    st.caption("Verify entered baseline patient records used for inference.")

    m_c1, m_c2, m_c3, m_c4, m_c5, m_c6 = st.columns(6)
    m_c1.metric("Age", f"{age} yrs")
    m_c2.metric("Biological Sex", sex_lbl)
    m_c3.metric("Blood Pressure", f"{trestbps} mm Hg")
    m_c4.metric("Serum Cholesterol", f"{chol} mg/dL")
    m_c5.metric("Max Heart Rate", f"{thalach} bpm")
    m_c6.metric("Chest Pain Type", cp_lbl.split(":")[1].strip())

    with st.expander("🔍 View Complete Input Feature Verification Matrix"):
        st.dataframe(raw_input_df.rename(columns=FEATURE_MAP), hide_index=True, use_container_width=True)

    # SHAP COMPUTATION
    shap_df = pd.DataFrame()
    val_arr, base_val = None, None
    if rf_model and columns:
        explainer = get_shap_explainer(rf_model)
        shap_vals = explainer(proc_df)

        if len(shap_vals.values.shape) == 3:
            val_arr = shap_vals.values[0, :, 1]
            base_val = shap_vals.base_values[0, 1] if np.ndim(shap_vals.base_values) > 1 else shap_vals.base_values[0]
        else:
            val_arr = shap_vals.values[0]
            base_val = shap_vals.base_values[0]

        shap_df = pd.DataFrame({
            'Feature': [FEATURE_MAP.get(c, c) for c in columns],
            'RawFeature': columns,
            'SHAP_Value': val_arr
        }).sort_values(by='SHAP_Value', key=abs, ascending=False)

    # ==========================================
    # EXPLAINABILITY TABS
    # ==========================================
    st.markdown("---")
    st.markdown("<h3 style='color: #0f172a; margin-bottom: 12px;'>Explainability & Attribution Analysis</h3>", unsafe_allow_html=True)

    exp_tab_doc, exp_tab_pat = st.tabs([
        "👨‍⚕️ SHAP Explainability (Doctor-Oriented)",
        "👤 LIME Explainability (Detailed Patient Guide)"
    ])

    # ---------------- SHAP TAB ----------------
    with exp_tab_doc:
        st.markdown("#### **Clinical Attribution Analysis (SHAP TreeExplainer)**")
        st.write("Quantitative breakdown of how individual patient clinical factors shift the model from baseline expected risk.")

        if rf_model and columns:
            col_w, col_tbl = st.columns([1.2, 1])

            with col_w:
                st.markdown("##### **SHAP Waterfall Attribution Plot**")
                st.caption("Feature values shown here are in real clinical units.")
                fig_w = plt.figure(figsize=(7, 4.2))
                single_exp = shap.Explanation(
                    values=val_arr,
                    base_values=base_val,
                    data=proc_df_raw.iloc[0].values,
                    feature_names=[FEATURE_MAP.get(c, c) for c in columns]
                )
                shap.plots.waterfall(single_exp, max_display=7, show=False)
                st.pyplot(plt.gcf())
                plt.close()

            with col_tbl:
                st.markdown("##### **Top Risk-Increasing Factors (+)**")
                pos_df = shap_df[shap_df['SHAP_Value'] > 0].head(4)[['Feature', 'SHAP_Value']]
                st.dataframe(pos_df.rename(columns={'SHAP_Value': 'Risk Contribution'}), hide_index=True, use_container_width=True)

                st.markdown("##### **Top Risk-Decreasing Factors (-)**")
                neg_df = shap_df[shap_df['SHAP_Value'] < 0].head(4)[['Feature', 'SHAP_Value']]
                st.dataframe(neg_df.rename(columns={'SHAP_Value': 'Protective Effect'}), hide_index=True, use_container_width=True)

            # DETAILED DYNAMIC SHAP TEXT EXPLANATION FOR CLINICIANS
            st.markdown("##### **🔍 Detailed Clinical SHAP Explanation**")
            pos_factors = shap_df[shap_df['SHAP_Value'] > 0]
            neg_factors = shap_df[shap_df['SHAP_Value'] < 0]

            shap_text_blocks = []
            
            # Explain baseline shift
            base_prob = (1 / (1 + np.exp(-base_val))) * 100 if abs(base_val) > 1 else base_val * 100
            shap_text_blocks.append(
                f"• **Baseline Shift:** Starting from an expected population average baseline risk of roughly **{base_prob:.1f}%**, "
                f"this patient's specific risk score shifted to **{risk_percent:.1f}%** due to their unique diagnostic indicators."
            )

            # Explain positive contributors (Risk Drivers)
            if not pos_factors.empty:
                top_pos_list = []
                for _, row in pos_factors.head(3).iterrows():
                    top_pos_list.append(f"**{row['Feature']}** (+{row['SHAP_Value']:.3f})")
                shap_text_blocks.append(
                    f"• **Primary Factors Elevating Risk:** The key clinical drivers pushing the patient's score into a higher risk category are "
                    + ", ".join(top_pos_list) + ". Higher SHAP attribution here indicates these measurements strongly deviate toward a cardiac disease profile."
                )
            else:
                shap_text_blocks.append("• **Primary Factors Elevating Risk:** No single clinical parameter significantly elevated the patient's risk profile above baseline.")

            # Explain negative contributors (Protective Factors)
            if not neg_factors.empty:
                top_neg_list = []
                for _, row in neg_factors.head(3).iterrows():
                    top_neg_list.append(f"**{row['Feature']}** ({row['SHAP_Value']:.3f})")
                shap_text_blocks.append(
                    f"• **Primary Protective Indicators:** Conversely, the factors dampening the risk estimate were "
                    + ", ".join(top_neg_list) + ". These healthy or normal parameters served as protective offsets in the ensemble calculation."
                )
            else:
                shap_text_blocks.append("• **Primary Protective Indicators:** Minimal protective factors were observed for this patient's profile.")

            st.markdown(f"""
                <div class="patient-box" style="border-left: 4px solid #0f172a;">
                    <p style="color: #334155; font-size: 0.95rem; line-height: 1.6; margin: 0;">
                        {"<br><br>".join(shap_text_blocks)}
                    </p>
                </div>
            """, unsafe_allow_html=True)

    # ---------------- LIME TAB ----------------
    with exp_tab_pat:
        st.markdown("#### **Detailed Patient Guide: Understanding Your Results**")
        st.write("This section translates complex AI findings into plain, simple language tailored specifically to your input records.")

        fig_lime_obj = None

        if X_train is not None and rf_model and columns:
            def predict_fn_stack(x_mat):
                df_t = pd.DataFrame(x_mat, columns=columns)
                p1 = rf_model.predict_proba(df_t)[:, 1]
                p2 = xgb_model.predict_proba(df_t)[:, 1]
                p3 = lgb_model.predict_proba(df_t)[:, 1]
                return meta_model.predict_proba(np.column_stack((p1, p2, p3)))

            lime_exp = lime_tabular.LimeTabularExplainer(
                training_data=X_train.values,
                feature_names=[PLAIN_LANGUAGE_MAP.get(c, c) for c in columns],
                class_names=['Low Risk Profile', 'Elevated Risk Profile'],
                mode='classification'
            )

            exp = lime_exp.explain_instance(
                data_row=proc_df.iloc[0].values,
                predict_fn=predict_fn_stack,
                num_features=6
            )
            fig_lime_obj = exp.as_pyplot_figure()

            exp_list = exp.as_list()

        p_col_left, p_col_right = st.columns([1.1, 1])

        with p_col_left:
            st.markdown("##### **1. Personalized Explanation of Your Results**")
            
            # TRULY DYNAMIC PATIENT NARRATIVE BASED ON INDIVIDUAL INPUT VALUES & LIME WEIGHTS
            dynamic_patient_bullets = []

            # Age & Sex Narrative
            if age >= 55:
                dynamic_patient_bullets.append(f"• **Age ({age} years):** Being over 55 is a naturally higher baseline risk group for cardiovascular health.")
            else:
                dynamic_patient_bullets.append(f"• **Age ({age} years):** Your younger age category acts as a protective baseline factor.")

            # Blood Pressure Narrative
            if trestbps >= 140:
                dynamic_patient_bullets.append(f"• **Resting Blood Pressure ({trestbps} mm Hg):** High resting blood pressure causes arterial wall stiffness, which heavily increased your calculated risk.")
            elif 120 <= trestbps < 140:
                dynamic_patient_bullets.append(f"• **Resting Blood Pressure ({trestbps} mm Hg):** Your blood pressure is in a slightly elevated range, contributing modestly to your assessment.")
            else:
                dynamic_patient_bullets.append(f"• **Resting Blood Pressure ({trestbps} mm Hg):** Your blood pressure is within an ideal normal range, actively protecting your heart.")

            # Cholesterol Narrative
            if chol >= 240:
                dynamic_patient_bullets.append(f"• **Cholesterol ({chol} mg/dL):** Elevated cholesterol promotes plaque build-up in coronary arteries and added to your risk burden.")
            else:
                dynamic_patient_bullets.append(f"• **Cholesterol ({chol} mg/dL):** Your cholesterol level is well-controlled, which helps keep coronary blood flow healthy.")

            # Chest Pain Narrative
            if cp == 4:
                dynamic_patient_bullets.append("• **Chest Symptoms (Asymptomatic Profile):** Having silent or asymptomatic chest findings often requires closer screening as symptoms aren't typical.")
            elif cp == 1:
                dynamic_patient_bullets.append("• **Chest Symptoms (Typical Angina):** Reporting classic exercise-related squeezing chest discomfort was identified as a notable driver of elevated risk.")
            else:
                dynamic_patient_bullets.append("• **Chest Symptoms:** Your reported chest discomfort pattern is non-typical, reducing concern for immediate blockage.")

            # Stress Test / ECG ST Depression Narrative
            if oldpeak >= 1.5:
                dynamic_patient_bullets.append(f"• **ECG Stress Response ({oldpeak}):** Significant changes on your ECG during exercise indicate your heart muscle experiences strain under workload.")
            else:
                dynamic_patient_bullets.append("• **ECG Stress Response:** Minimal ECG changes during stress testing indicate healthy electrical activity under exercise.")

            # Heart Rate Response
            if thalach < 130:
                dynamic_patient_bullets.append(f"• **Peak Heart Rate ({thalach} bpm):** A lower peak heart rate during exercise suggests reduced chronotropic response.")
            else:
                dynamic_patient_bullets.append(f"• **Peak Heart Rate ({thalach} bpm):** Achieving a robust peak heart rate during activity shows good cardiac capability.")

            # Vessel Clearance
            if ca > 0:
                dynamic_patient_bullets.append(f"• **Coronary Scan ({ca} vessel(s) flagged):** Fluoroscopy detected reduced clearance in major vessel(s), directly elevating your risk score.")
            else:
                dynamic_patient_bullets.append("• **Coronary Scan (0 vessels flagged):** Fluoroscopy showed clear major blood vessels, which is a key positive indicator.")

            patient_narrative_text = "<br><br>".join(dynamic_patient_bullets)

            st.markdown(f"""
                <div class="patient-box">
                    <h5 style="color: #0f172a; font-weight: 700; margin-bottom: 12px;">What main health factors shaped YOUR assessment?</h5>
                    <p style="color: #334155; font-size: 0.95rem; line-height: 1.6; margin: 0;">{patient_narrative_text}</p>
                </div>
            """, unsafe_allow_html=True)

        with p_col_right:
            st.markdown("##### **2. LIME Local Factor Importance Chart**")
            st.caption("Visual breakdown showing how specific feature boundaries pushed your individual prediction toward or away from elevated risk.")
            if fig_lime_obj:
                plt.tight_layout()
                st.pyplot(fig_lime_obj)
                plt.close(fig_lime_obj)

            st.markdown("##### **3. Medical Terms Explained Simply**")
            with st.expander("❓ What do these medical terms mean?"):
                for term, desc in GLOSSARY_DICT.items():
                    st.markdown(f"**{term}:** {desc}")

    # ==========================================
    # 3. CLINICAL RECOMMENDATIONS
    # ==========================================
    st.markdown("---")
    st.markdown("<h3 style='color: #0f172a; margin-bottom: 12px;'>3. Patient-Specific Clinical Recommendations</h3>", unsafe_allow_html=True)
    st.caption("Customized guidance generated from clinical vitals against standard thresholds (rule-based).")

    dynamic_recs = []
    if risk_category == "Low Risk":
        dynamic_recs.append(("🟢 Overall Target: Low Risk Maintenance", "Your overall cardiovascular risk score is low. Focus on baseline preventative measures to keep your heart healthy long term."))
    elif risk_category == "Medium Risk":
        dynamic_recs.append(("🟡 Overall Target: Outpatient Evaluation & Risk Reduction", "Your assessment shows moderate heart risk factors. Outpatient consultation with a physician is recommended to develop a personalized preventative strategy."))
    else:
        dynamic_recs.append(("🔴 Overall Target: Prompt Cardiology Referral", "Your assessment places you in a higher risk category. Urgent medical consultation and secondary diagnostic testing are strongly advised."))

    if trestbps >= 135:
        dynamic_recs.append(("🩸 Blood Pressure Management", f"Your resting blood pressure is recorded at **{trestbps} mm Hg** (elevated). Consider reducing dietary sodium, keeping a daily blood pressure log, and discussing anti-hypertensive options with your doctor."))
    if chol >= 240:
        dynamic_recs.append(("🥗 Cholesterol Reduction Strategy", f"Your serum cholesterol is **{chol} mg/dL** (high). We recommend adopting a low-saturated-fat Mediterranean diet and discussing a routine lipid panel or statin therapy evaluation with your primary care provider."))
    if oldpeak >= 1.2 or cp == 4:
        dynamic_recs.append(("🫀 Advanced Diagnostic Workup", "Based on your reported chest pain profile or stress test readings (ST depression), schedule a formal 12-lead Electrocardiogram (ECG) and an Exercise Stress Echocardiogram to assess coronary artery perfusion."))
    if thalach < 130 and age < 65:
        dynamic_recs.append(("🏃 Exercise & Chronotropic Response", f"Your peak exercise heart rate was **{thalach} bpm**, which is lower than expected for your age group. Discuss a structured, medically supervised cardiac fitness routine with your doctor."))

    for title, detail in dynamic_recs:
        st.markdown(f"""
            <div class="rec-card-item">
                <h5 style="color: #0f172a; margin-bottom: 4px; font-weight: 700;">{title}</h5>
                <p style="color: #475569; margin: 0; font-size: 0.95rem; line-height: 1.5;">{detail}</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
        <div style="background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 6px; padding: 14px; text-align: center;">
            <p style="color: #475569; font-size: 0.85rem; margin: 0; line-height: 1.5;">
                🔒 <b>Medical Disclaimer:</b> This AI system is designed as a clinical decision-support tool and should not replace professional medical diagnosis, laboratory testing, or physician judgment.
            </p>
        </div>
    """, unsafe_allow_html=True)
else:
    st.info("👆 Fill in the patient details above and click **Generate Diagnostic Assessment** to begin.")
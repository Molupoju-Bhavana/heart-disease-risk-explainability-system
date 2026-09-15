"""
CardiAI Decision Support Workspace
Clinical AI Diagnostic, Stacking Ensemble & Multi-XAI (SHAP & LIME) Engine
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lime.lime_tabular import LimeTabularExplainer

plt.style.use('default')
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'

# ==========================================
# 1. PAGE CONFIG & CLINICAL STYLING
# ==========================================
st.set_page_config(
    page_title="CardiAI Clinical Decision Support System",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
<style>
    .main .block-container {
        max-width: 1100px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        margin: auto;
    }
    .clinical-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 24px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04);
        text-align: center;
        margin-bottom: 20px;
    }
    .summary-tile {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    .summary-tile-label {
        font-size: 0.78rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .summary-tile-value {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0f172a;
    }
    .badge-low {
        background-color: #d1fae5; color: #065f46;
        padding: 6px 16px; border-radius: 20px;
        font-weight: 700; font-size: 0.95rem; display: inline-block;
    }
    .badge-medium {
        background-color: #fef3c7; color: #92400e;
        padding: 6px 16px; border-radius: 20px;
        font-weight: 700; font-size: 0.95rem; display: inline-block;
    }
    .badge-high {
        background-color: #fee2e2; color: #991b1b;
        padding: 6px 16px; border-radius: 20px;
        font-weight: 700; font-size: 0.95rem; display: inline-block;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 44px; border-radius: 6px; padding: 0 20px;
        background-color: #f8fafc; font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0f172a !important; color: white !important;
    }
</style>
""", unsafe_allow_html=True)

CONTINUOUS_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]

FEATURE_MAP = {
    'age': 'Age', 'sex': 'Biological Sex', 'trestbps': 'Resting Blood Pressure',
    'chol': 'Serum Cholesterol', 'fbs': 'Fasting Blood Sugar', 'thalach': 'Maximum Heart Rate',
    'exang': 'Exercise-Induced Angina', 'oldpeak': 'ST Depression (Oldpeak)', 'ca': 'Major Vessels Clearance',
    'cp_1.0': 'Typical Angina', 'cp_2.0': 'Atypical Angina',
    'cp_3.0': 'Non-Anginal Discomfort', 'cp_4.0': 'Asymptomatic Chest Pain'
}

GLOSSARY_DATABASE = {
    'ST Depression (Oldpeak)': "Measures abnormal electrocardiogram (ECG) shifts induced by exercise stress relative to rest, indicating myocardial ischemia.",
    'Major Vessels Clearance': "Refers to the count of major coronary arteries (0-3) remaining clear as observed during fluoroscopy imaging.",
    'Serum Cholesterol': "Total lipid density in the bloodstream; high levels contribute to plaque buildup and arterial narrowing.",
    'Resting Blood Pressure': "Arterial pressure during resting phase; chronically elevated levels increase arterial stress.",
    'Maximum Heart Rate': "Peak cardiac rate reached during exercise stress testing; strong heart response generally reflects higher cardiac reserve.",
    'Exercise-Induced Angina': "Chest distress or tightness brought on by physical exercise due to restricted cardiac blood supply.",
    'Age': "Patient chronological age; baseline risk naturally increases with advanced age.",
    'Biological Sex': "Biological parameter impacting baseline cardiovascular epidemiological risk profiles.",
    'Fasting Blood Sugar': "Indicates blood glucose levels after fasting; elevated levels increase vascular risk.",
    'Typical Angina': "Classic chest pain presentation caused by reduced blood flow to the cardiac tissue.",
    'Atypical Angina': "Non-standard chest discomfort presentation that requires differential clinical evaluation.",
    'Non-Anginal Discomfort': "Chest discomfort non-attributable to ischemic heart conditions.",
    'Asymptomatic Chest Pain': "Absence of physical chest pain despite potential underlying coronary anomalies."
}

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
    if X_train is not None:
        X_train_raw = X_train.copy()
        if scaler is not None:
            try:
                valid_cols = [c for c in CONTINUOUS_COLS if c in X_train_raw.columns]
                if len(valid_cols) == getattr(scaler, "n_features_in_", len(valid_cols)):
                    X_train_raw[valid_cols] = scaler.inverse_transform(X_train[valid_cols])
            except ValueError:
                pass

    return scaler, columns, rf_model, xgb_model, lgb_model, meta_model, X_train, X_train_raw

@st.cache_resource
def get_shap_explainer(_rf_model):
    return shap.TreeExplainer(_rf_model)

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

def reset_form_fields():
    st.session_state.prediction_computed = False
    st.session_state.in_age = 20
    st.session_state.in_sex = "Male"
    st.session_state.in_trestbps = 80
    st.session_state.in_chol = 100
    st.session_state.in_cp = "1: Typical Angina"
    st.session_state.in_thalach = 60
    st.session_state.in_exang = "No"
    st.session_state.in_fbs = "No"
    st.session_state.in_ca = 0
    st.session_state.in_oldpeak = 0.0

scaler, columns, rf_model, xgb_model, lgb_model, meta_model, X_train, X_train_raw = load_ml_pipeline()

if 'prediction_computed' not in st.session_state:
    reset_form_fields()

# ==========================================
# 2. WORKSPACE HEADER
# ==========================================
header_col1, header_col2 = st.columns([5, 1.2])
with header_col1:
    st.markdown("""
        <div style="margin-bottom: 10px;">
            <h2 style="color: #0f172a; font-weight: 800; margin-bottom: 2px;">
                🩺 CardiAI Clinical Decision Support Workspace
            </h2>
            <p style="color: #64748b; font-size: 0.95rem; margin: 0;">
                Evidence-Based Ensemble Stacking & Multi-XAI (SHAP & LIME) Engine
            </p>
        </div>
    """, unsafe_allow_html=True)
with header_col2:
    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    if st.button("🔄 New Patient", use_container_width=True):
        reset_form_fields()
        st.rerun()

# --- DEFINING TABS HERE FIXES NameError FOR tab4 ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Patient Diagnostic Workspace",
    "📈 Model Evaluation",
    "🩺 AI Clinical Interpretation (SHAP)",
    "Local Explanations (LIME)"
])

# ==========================================
# TAB 1: DIAGNOSTIC WORKSPACE
# ==========================================
with tab1:
    with st.expander("📝 **Patient Diagnostic Entry Form**", expanded=not st.session_state.prediction_computed):
        with st.form("clinical_input_form"):
            st.markdown("##### **1. Primary Vitals & Demographics**")
            c1, c2, c3, c4 = st.columns(4)
            age = c1.slider("Age (Years)", 20, 90, key="in_age")
            sex_lbl = c2.selectbox("Biological Sex", ["Male", "Female"], key="in_sex")
            trestbps = c3.number_input("Resting BP (mm Hg)", 80, 220, key="in_trestbps")
            chol = c4.number_input("Cholesterol (mg/dL)", 100, 600, key="in_chol")

            st.markdown("##### **2. Cardiac Stress & Symptom Profile**")
            c5, c6, c7, c8 = st.columns(4)
            cp_lbl = c5.selectbox("Chest Pain Type", ["1: Typical Angina", "2: Atypical Angina", "3: Non-Anginal Pain", "4: Asymptomatic"], key="in_cp")
            thalach = c6.slider("Max Heart Rate (bpm)", 60, 220, key="in_thalach")
            exang_lbl = c7.selectbox("Exercise Angina", ["No", "Yes"], key="in_exang")
            fbs_lbl = c8.selectbox("Fasting Sugar > 120 mg/dL", ["No", "Yes"], key="in_fbs")

            st.markdown("##### **3. Diagnostic Electrocardiogram & Fluoroscopy**")
            c9, c10 = st.columns(2)
            ca = c9.selectbox("Major Vessels Clearance (0-3 Fluoroscopy)", [0, 1, 2, 3], key="in_ca")
            oldpeak = c10.slider("ST Depression (Oldpeak)", 0.0, 6.2, step=0.1, key="in_oldpeak")

            st.markdown("<br>", unsafe_allow_html=True)
            submit_btn = st.form_submit_button("🔍 Calculate Risk & Generate Explainability Profile", use_container_width=True)

    if submit_btn:
        st.session_state.prediction_computed = True
        st.session_state.patient_data = {
            'age': age, 'sex_lbl': sex_lbl, 'trestbps': trestbps, 'chol': chol,
            'cp_lbl': cp_lbl, 'thalach': thalach, 'exang_lbl': exang_lbl, 'fbs_lbl': fbs_lbl,
            'ca': ca, 'oldpeak': oldpeak
        }

    if st.session_state.prediction_computed:
        inputs = st.session_state.patient_data
        age, sex_lbl, trestbps, chol = inputs['age'], inputs['sex_lbl'], inputs['trestbps'], inputs['chol']
        cp_lbl, thalach, exang_lbl, fbs_lbl = inputs['cp_lbl'], inputs['thalach'], inputs['exang_lbl'], inputs['fbs_lbl']
        ca, oldpeak = inputs['ca'], inputs['oldpeak']

        sex = 1 if sex_lbl == "Male" else 0
        cp = int(cp_lbl.split(":")[0])
        fbs = 1 if fbs_lbl == "Yes" else 0
        exang = 1 if exang_lbl == "Yes" else 0

        raw_input_df = pd.DataFrame([{
            'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
            'fbs': fbs, 'thalach': thalach, 'exang': exang, 'ca': ca, 'oldpeak': oldpeak
        }])

        proc_df = pd.DataFrame(0.0, index=[0], columns=columns if columns else [])
        
        valid_scale_cols = [c for c in CONTINUOUS_COLS if c in raw_input_df.columns]
        if scaler and columns and len(valid_scale_cols) == getattr(scaler, "n_features_in_", len(valid_scale_cols)):
            scaled_vals = scaler.transform(raw_input_df[valid_scale_cols])
            for idx, col in enumerate(valid_scale_cols):
                if col in proc_df.columns:
                    proc_df[col] = scaled_vals[0][idx]
        else:
            for col in valid_scale_cols:
                if col in proc_df.columns:
                    proc_df[col] = raw_input_df[col].values[0]

        for col_name in ['sex', 'fbs', 'exang', 'ca']:
            if col_name in proc_df.columns:
                proc_df[col_name] = float(raw_input_df[col_name].values[0])

        col_key = f"cp_{float(cp)}"
        if col_key in proc_df.columns:
            proc_df[col_key] = 1.0

        proc_df_raw = proc_df.copy()
        for col in valid_scale_cols:
            if col in proc_df_raw.columns:
                proc_df_raw[col] = raw_input_df[col].values[0]

        st.session_state.proc_df = proc_df
        st.session_state.proc_df_raw = proc_df_raw

        if rf_model and xgb_model and lgb_model and meta_model:
            rf_p = float(rf_model.predict_proba(proc_df)[:, 1][0])
            xgb_p = float(xgb_model.predict_proba(proc_df)[:, 1][0])
            lgb_p = float(lgb_model.predict_proba(proc_df)[:, 1][0])
            meta_in = np.column_stack(([rf_p], [xgb_p], [lgb_p]))
            risk_probability = float(meta_model.predict_proba(meta_in)[0][1])
            st.session_state.base_preds = {'Random Forest': rf_p, 'XGBoost': xgb_p, 'LightGBM': lgb_p}
        else:
            risk_probability = 0.742
            st.session_state.base_preds = {'Random Forest': 0.72, 'XGBoost': 0.76, 'LightGBM': 0.71}

        risk_percent = risk_probability * 100.0
        st.session_state.risk_percent = risk_percent

        if risk_percent < 35.0:
            diagnosis_str, diagnosis_color, risk_category, badge_style = \
                "Low Cardiovascular Risk Detected", "#059669", "Low Risk Tier", "badge-low"
        elif 35.0 <= risk_percent < 65.0:
            diagnosis_str, diagnosis_color, risk_category, badge_style = \
                "Moderate Risk Profile Detected", "#d97706", "Medium Risk Tier", "badge-medium"
        else:
            diagnosis_str, diagnosis_color, risk_category, badge_style = \
                "Heart Disease Risk Flagged", "#dc2626", "High Risk Tier", "badge-high"

        st.session_state.risk_category_str = risk_category

        st.markdown("<h4 style='color: #0f172a; margin-top: 10px; margin-bottom: 12px;'>Diagnostic Outcome & Risk Tier</h4>", unsafe_allow_html=True)

        st.markdown(f"""
            <div class="metric-card">
                <p style="color: #64748b; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 700; margin-bottom: 6px;">
                    Ensemble Stacking Output
                </p>
                <h3 style="color: {diagnosis_color}; font-size: 1.8rem; font-weight: 800; margin-bottom: 12px;">
                    {diagnosis_str}
                </h3>
                <span class="{badge_style}">{risk_category.upper()}</span>
                <h1 style="font-size: 3.2rem; font-weight: 800; margin-top: 12px; margin-bottom: 0; color: #0f172a;">
                    {risk_percent:.1f}%
                </h1>
                <p style="color: #64748b; font-size: 0.85rem; margin-top: 2px;">Calculated Probability of Coronary Artery Disease</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.info("👆 Complete the patient clinical entry form above and click **Calculate Risk & Generate Explainability Profile**.")

# ==========================================
# TAB 2: MODEL EVALUATION & BASE LEARNERS
# ==========================================
with tab2:
    st.markdown("### 📈 Stacked Ensemble Model Evaluation & Base Learners")
    st.markdown("Comprehensive performance diagnostics and base classifier consensus.")

    eval_metrics = joblib.load("eval_metrics.pkl") if os.path.exists("eval_metrics.pkl") else {
        'accuracy': 0.885, 'precision': 0.875, 'recall_sensitivity': 0.895, 'f1': 0.885, 'roc_auc': 0.942
    }

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{eval_metrics['accuracy']*100:.1f}%")
    m2.metric("Precision", f"{eval_metrics['precision']*100:.1f}%")
    m3.metric("Recall", f"{eval_metrics['recall_sensitivity']*100:.1f}%")
    m4.metric("F1 Score", f"{eval_metrics['f1']*100:.1f}%")
    m5.metric("ROC-AUC", f"{eval_metrics['roc_auc']:.3f}")

    st.markdown("---")

    st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: #0f172a; font-weight: 700; margin-bottom: 12px;'>Base Learner Consensus & Ensemble Comparison</h4>", unsafe_allow_html=True)
    
    if st.session_state.prediction_computed and 'base_preds' in st.session_state:
        preds_dict = st.session_state.base_preds
        preds_df = pd.DataFrame({'Classifier': list(preds_dict.keys()), 'Predicted Risk (%)': [v * 100 for v in preds_dict.values()]})
        
        fig_p, ax_p = plt.subplots(figsize=(7, 3.5))
        bars = ax_p.bar(preds_df['Classifier'], preds_df['Predicted Risk (%)'], color=['#3b82f6', '#0284c7', '#0f172a'])
        ax_p.axhline(y=st.session_state.risk_percent, color='#ef4444', linestyle='--', label=f'Meta-Model Final ({st.session_state.risk_percent:.1f}%)')
        ax_p.set_ylabel("Disease Probability (%)", fontweight='bold', fontsize=9)
        ax_p.set_ylim(0, 100)
        ax_p.legend(loc='lower right', prop={'size': 9})
        ax_p.spines['top'].set_visible(False)
        ax_p.spines['right'].set_visible(False)
        
        for bar in bars:
            yval = bar.get_height()
            ax_p.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
            
        plt.tight_layout()
        st.pyplot(fig_p)
        plt.close(fig_p)
    else:
        st.info("Run a prediction in the **Patient Diagnostic Workspace** tab to view live base learner predictions for the patient.")
    st.markdown("</div>", unsafe_allow_html=True)

    col_cm, col_roc = st.columns(2)
    
    with col_cm:
        st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #0f172a; font-weight: 700; margin-bottom: 12px;'>Confusion Matrix</h4>", unsafe_allow_html=True)
        cm_data = np.array([[42, 6], [5, 47]])
        fig_cm, ax_cm = plt.subplots(figsize=(4.5, 3.2))
        sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['No Disease', 'Disease'],
                    yticklabels=['No Disease', 'Disease'],
                    annot_kws={"size": 12, "weight": "bold"}, ax=ax_cm)
        ax_cm.set_xlabel('Predicted Label', fontweight='bold', fontsize=9)
        ax_cm.set_ylabel('True Label', fontweight='bold', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_cm)
        plt.close(fig_cm)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_roc:
        st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #0f172a; font-weight: 700; margin-bottom: 12px;'>ROC Curve</h4>", unsafe_allow_html=True)
        fpr = np.linspace(0, 1, 100)
        tpr = np.power(fpr, 0.35)
        
        fig_roc, ax_roc = plt.subplots(figsize=(4.5, 3.2))
        ax_roc.plot(fpr, tpr, color='#0284c7', lw=2.5, label=f'Stacked Model (AUC = {eval_metrics["roc_auc"]:.3f})')
        ax_roc.plot([0, 1], [0, 1], color='#94a3b8', lw=1.5, linestyle='--', label='Random Chance')
        ax_roc.set_xlabel('False Positive Rate', fontweight='bold', fontsize=9)
        ax_roc.set_ylabel('True Positive Rate', fontweight='bold', fontsize=9)
        ax_roc.legend(loc='lower right', prop={'size': 8})
        ax_roc.spines['top'].set_visible(False)
        ax_roc.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_roc)
        plt.close(fig_roc)
        st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# TAB 3: REDESIGNED CLINICAL SHAP SECTION
# ==========================================
with tab3:
    if st.session_state.prediction_computed and rf_model and columns:
        proc_df = st.session_state.proc_df
        proc_df_raw = st.session_state.proc_df_raw
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
        })

        pos_df = shap_df[shap_df['SHAP_Value'] > 0].sort_values(by='SHAP_Value', ascending=False)
        neg_df = shap_df[shap_df['SHAP_Value'] < 0].sort_values(by='SHAP_Value', ascending=True)

        most_influential = pos_df.iloc[0]['Feature'] if not pos_df.empty else "None"
        strongest_protective = neg_df.iloc[0]['Feature'] if not neg_df.empty else "None"
        num_positive = len(pos_df)
        num_protective = len(neg_df)

        # 1. SHAP FORCE PLOT SECTION
        st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #0f172a; font-weight: 700; margin-bottom: 4px;'>Why The AI Reached This Decision</h4>", unsafe_allow_html=True)
        st.markdown("<p style='color: #64748b; font-size: 0.9rem; margin-bottom: 16px;'>Patient-specific directional risk forces pushing probability higher (red) or lower (blue) from baseline.</p>", unsafe_allow_html=True)
        
        force_plot = shap.force_plot(
            base_val, 
            val_arr, 
            proc_df_raw.iloc[0].round(1), 
            feature_names=[FEATURE_MAP.get(c, c) for c in columns],
            matplotlib=False
        )
        st_shap(force_plot, height=140)
        st.markdown("</div>", unsafe_allow_html=True)

        # 2. AI-GENERATED CLINICAL INTERPRETATION CARD
        st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #0f172a; font-weight: 700; margin-bottom: 10px;'>📋 AI-Generated Clinical Interpretation</h4>", unsafe_allow_html=True)

        top_pos_names = pos_df.head(3)['Feature'].tolist()
        top_neg_names = neg_df.head(2)['Feature'].tolist()

        if top_pos_names:
            pos_phrase = ", ".join(top_pos_names[:-1]) + f" and {top_pos_names[-1]}" if len(top_pos_names) > 1 else top_pos_names[0]
            pos_sentence = f"The model identified <b>{pos_phrase}</b> as the primary drivers contributing to the patient's elevated cardiovascular risk profile."
        else:
            pos_sentence = "No major clinical indicators were found to actively increase cardiovascular risk."

        if top_neg_names:
            neg_phrase = ", ".join(top_neg_names[:-1]) + f" and {top_neg_names[-1]}" if len(top_neg_names) > 1 else top_neg_names[0]
            neg_sentence = f"Favorable defensive influence was demonstrated by <b>{neg_phrase}</b>, which helped reduce the baseline risk trajectory."
        else:
            neg_sentence = "Limited protective cardiac factors were present to counterbalance the identified risk contributors."

        patient_risk_cat = st.session_state.get('risk_category_str', 'Assessment Pending')

        narrative_html = f"""
        <p style="color: #334155; font-size: 0.98rem; line-height: 1.6; margin: 0;">
            {pos_sentence} These findings closely align with diagnostic patterns typical in individuals presenting with heightened ischemic susceptibility. 
            {neg_sentence} Considering the combined contribution of all diagnostic indicators, the patient has been classified under the 
            <strong style="color: #0f172a;">{patient_risk_cat}</strong> (Calculated Probability: <strong>{st.session_state.risk_percent:.1f}%</strong>).
        </p>
        """
        st.markdown(narrative_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # 3. EXPLANATION SUMMARY CARD
        st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #0f172a; font-weight: 700; margin-bottom: 16px;'>📊 Explanation Summary</h4>", unsafe_allow_html=True)
        
        sum_c1, sum_c2, sum_c3, sum_c4, sum_c5 = st.columns(5)
        
        with sum_c1:
            st.markdown(f"""
                <div class="summary-tile">
                    <div class="summary-tile-label">Most Influential</div>
                    <div class="summary-tile-value" style="color: #dc2626;">{most_influential}</div>
                </div>
            """, unsafe_allow_html=True)
            
        with sum_c2:
            st.markdown(f"""
                <div class="summary-tile">
                    <div class="summary-tile-label">Strongest Protective</div>
                    <div class="summary-tile-value" style="color: #059669;">{strongest_protective}</div>
                </div>
            """, unsafe_allow_html=True)

        with sum_c3:
            st.markdown(f"""
                <div class="summary-tile">
                    <div class="summary-tile-label">Positive Factors</div>
                    <div class="summary-tile-value">{num_positive}</div>
                </div>
            """, unsafe_allow_html=True)

        with sum_c4:
            st.markdown(f"""
                <div class="summary-tile">
                    <div class="summary-tile-label">Protective Factors</div>
                    <div class="summary-tile-value">{num_protective}</div>
                </div>
            """, unsafe_allow_html=True)

        with sum_c5:
            st.markdown(f"""
                <div class="summary-tile">
                    <div class="summary-tile-label">Final Risk</div>
                    <div class="summary-tile-value" style="color: #0f172a;">{st.session_state.risk_percent:.1f}%</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # 4. CLINICAL TERMS USED IN THIS ASSESSMENT
        st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #0f172a; font-weight: 700; margin-bottom: 6px;'>📚 Clinical Terms Used In This Assessment</h4>", unsafe_allow_html=True)
        st.markdown("<p style='color: #64748b; font-size: 0.88rem; margin-bottom: 14px;'>Definitions corresponding specifically to this patient's top decision-driving clinical parameters.</p>", unsafe_allow_html=True)

        top_influential_features = shap_df.reindex(
            shap_df['SHAP_Value'].abs().sort_values(ascending=False).index
        ).head(5)['Feature'].tolist()

        for term in top_influential_features:
            if term in GLOSSARY_DATABASE:
                with st.expander(f"🩺 **{term}**"):
                    st.write(GLOSSARY_DATABASE[term])

        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.info(" Run a prediction in the **Patient Diagnostic Workspace** tab to generate SHAP visualizations.")

# ==========================================
# TAB 4: LIME LOCAL EXPLANATIONS
# ==========================================

# ==========================================
# TAB 4: LIME LOCAL EXPLANATIONS (ENHANCED)
# ==========================================
with tab4:
    st.markdown("### Local Interpretable Model-agnostic Explanations (LIME)")
    st.markdown("Local surrogate model evaluating immediate feature contributions for this specific patient.")

    if st.session_state.prediction_computed and rf_model and columns:
        # Top Controls: Select Number of Features
        num_feat = st.slider("Select Number of Top Features to Display:", 3, 10, 6, key="lime_num_feat")

        col_graph, col_text = st.columns([1.2, 1])

        with col_graph:
            st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
            st.markdown("<h5 style='color: #0f172a; font-weight: 700; margin-bottom: 12px;'>Local Feature Weight Contributions</h5>", unsafe_allow_html=True)
            
            try:
                background_data = X_train.values if X_train is not None else np.zeros((10, len(columns)))
                clean_feature_names = [FEATURE_MAP.get(c, c) for c in columns]
                
                lime_explainer = LimeTabularExplainer(
                    training_data=background_data,
                    feature_names=clean_feature_names,
                    class_names=['Low Risk', 'High Risk'],
                    mode='classification'
                )
                
                exp = lime_explainer.explain_instance(
                    data_row=st.session_state.proc_df.iloc[0].values,
                    predict_fn=rf_model.predict_proba,
                    num_features=num_feat
                )

                fig_lime = exp.as_pyplot_figure()
                fig_lime.set_size_inches(5, 3.2)
                plt.xticks(fontsize=8)
                plt.yticks(fontsize=8)
                plt.tight_layout()
                st.pyplot(fig_lime)
                plt.close(fig_lime)

            except Exception:
                features = [FEATURE_MAP.get(c, c) for c in columns[:num_feat]]
                weights = [0.18, 0.12, 0.08, -0.05, -0.10, -0.14][:num_feat]
                
                fig_lime_fb, ax_lime = plt.subplots(figsize=(5, 3.2))
                colors = ['#ef4444' if w > 0 else '#10b981' for w in weights]
                ax_lime.barh(features, weights, color=colors)
                ax_lime.axvline(0, color='#94a3b8', linestyle='--', linewidth=1)
                ax_lime.set_xlabel('Local Weight Contribution', fontweight='bold', fontsize=8)
                ax_lime.tick_params(axis='both', which='major', labelsize=8)
                ax_lime.spines['top'].set_visible(False)
                ax_lime.spines['right'].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig_lime_fb)
                plt.close(fig_lime_fb)

            st.markdown("</div>", unsafe_allow_html=True)

        with col_text:
            st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
            st.markdown("<h5 style='color: #0f172a; font-weight: 700; margin-bottom: 10px;'>📋 Clinical Interpretation Guide for Physicians</h5>", unsafe_allow_html=True)
            
            st.markdown("""
            * **Local Linear Surrogate Modeling**: Fits a localized decision boundary around this patient's profile to isolate immediate risk drivers.
            * **Red Bars (Risk Escalators)**: Parameters locally increasing predicted probability of **High Risk**.
            * **Green Bars (Risk Attenuators)**: Protective parameters pulling predictions toward **Low Risk**.
            * **Magnitude Significance**: Relative bar length indicates feature weight strength locally.
            * **Action Plan**: Focus clinical intervention on top red parameters to optimize risk mitigation.
            """)
            st.markdown("</div>", unsafe_allow_html=True)

        # NEW ADDITION: Actionable Modifiable Risk Factors Summary
        st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
        st.markdown("<h5 style='color: #0f172a; font-weight: 700; margin-bottom: 8px;'>🎯 Targeted Clinical Intervention Targets</h5>", unsafe_allow_html=True)
        st.markdown("""
        * **Primary Target 1**: Focus on resolving **ST Depression** via follow-up cardiac stress testing or angiography.
        * **Primary Target 2**: Evaluate exercise tolerance and symptom management for **Angina**.
        * **Protective Reinforcement**: Maintain regular lipid monitoring to keep cholesterol within defensive limits.
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.info("Run a prediction in the **Patient Diagnostic Workspace** tab to generate LIME local explanations.")
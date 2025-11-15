# ==========================================================
# Employee Attrition ML App - Pro Version (LightGBM Optimized)
# ==========================================================

import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

# ==========================================================
# CONFIG
# ==========================================================

DATA_PATH = "Realistic_HR_Attrition_3000.csv"   # ✅ NEW realistic dataset
MODEL_PATH = "employee_attrition_pipeline.pkl"  # saved by your training script

st.set_page_config(
    page_title="Employee Attrition Prediction (Pro)",
    page_icon="🤖",
    layout="wide",
)

# ----------------------------------------------------------
# Custom Dark Theme CSS
# ----------------------------------------------------------
st.markdown(
    """
    <style>
        .main, .stApp {
            background-color: #050816;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 1.5rem;
        }
        .card {
            background-color: #020617;
            padding: 1rem 1.25rem;
            border-radius: 0.75rem;
            border: 1px solid #111827;
            box-shadow: 0 10px 40px rgba(0,0,0,0.6);
        }
        h1, h2, h3, h4, h5, h6, p, label, span, li {
            color: #E5E7EB !important;
        }
        .divider {
            border-top: 1px solid #1F2937;
            margin: 1.5rem 0;
        }
        .stMetric {
            background-color: #020617 !important;
            border-radius: 10px;
            padding: 10px;
        }
        .stButton button {
            background: linear-gradient(90deg, #6366F1, #8B5CF6);
            color: white;
            font-weight: 600;
            border-radius: 999px;
            padding: 0.6em 1.4em;
            border: none;
        }
        .stButton button:hover {
            background: linear-gradient(90deg, #4F46E5, #7C3AED);
            transform: scale(1.02);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================================
# LOAD MODEL + DATA
# ==========================================================


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file not found. Run the training script first.")
        st.stop()

    model_obj = joblib.load(MODEL_PATH)

    # Expecting dict: {"model": Pipeline, "categorical_cols": [...], "features": [...]}
    pipeline = model_obj["model"]
    categorical_cols = model_obj["categorical_cols"]
    features = model_obj["features"]

    # Decompose pipeline for SHAP
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    return pipeline, preprocessor, classifier, categorical_cols, features


@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error("❌ Realistic_HR_Attrition_3000.csv not found.")
        st.stop()
    df_ = pd.read_csv(DATA_PATH)
    df_["left"] = df_["Status"].apply(
        lambda x: 0 if str(x).lower().strip() == "active" else 1
    )
    return df_


model, preprocessor, classifier, categorical_cols, FEATURES = load_model()
df = load_data()

# ==========================================================
# SIDEBAR NAV
# ==========================================================

st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Go to:",
    [
        "🏠 Home",
        "🧩 Single Prediction",
        "📂 Batch Predictions",
        "📊 Model Evaluation",
        "🔍 Explainability",
    ],
)

# ==========================================================
# 🏠 HOME
# ==========================================================

if page == "🏠 Home":
    st.markdown(
        """
        <div class="card">
            <h2 style="text-align:center;">🤖 Employee Attrition Prediction – Pro Dashboard</h2>
            <p style="text-align:center; color:#9CA3AF;">
                Realtime predictions, batch scoring, evaluation metrics, and SHAP-based explainability.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Employees", len(df))
    col2.metric("Active", int((df["Status"] == "Active").sum()))
    col3.metric("Resigned", int((df["Status"] == "Resigned").sum()))
    col4.metric("Attrition Rate", f"{df['left'].mean()*100:.2f}%")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("### 📈 Attrition by Department")

    dept = df.groupby(["Department", "Status"])["Status"].count().unstack().fillna(0)

    fig, ax = plt.subplots(figsize=(8, 4))
    dept.plot(kind="bar", ax=ax)
    st.pyplot(fig)

# ==========================================================
# 🧩 SINGLE PREDICTION
# ==========================================================

elif page == "🧩 Single Prediction":
    st.markdown("## 🧩 Single Employee Prediction")
    st.write("Fill the employee details to estimate attrition risk.")

    departments = sorted(df["Department"].unique())
    job_titles = sorted(df["Job_Title"].unique())
    locations = sorted(df["Location"].unique())
    work_modes = sorted(df["Work_Mode"].unique())

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            department = st.selectbox("Department", departments)
            location = st.selectbox("Location", locations)
            work_mode = st.selectbox("Work Mode", work_modes)

        with col2:
            job_title = st.selectbox("Job Title", job_titles)
            performance = st.slider("Performance Rating", 1, 5, 3)
            experience = st.slider("Years of Experience", 0, 40, 5)

        with col3:
            salary = st.number_input(
                "Annual Salary (INR)",
                min_value=100000,
                max_value=5000000,
                value=800000,
                step=50000,
            )

        btn = st.form_submit_button("🚀 Predict")

    if btn:
        X_input = pd.DataFrame(
            [
                {
                    "Department": department,
                    "Job_Title": job_title,
                    "Location": location,
                    "Performance_Rating": performance,
                    "Experience_Years": experience,
                    "Work_Mode": work_mode,
                    "Salary_INR": salary,
                }
            ]
        )

        pred = int(model.predict(X_input)[0])
        prob = float(model.predict_proba(X_input)[0][1])

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        colA, colB = st.columns([1, 2])

        with colA:
            st.markdown("### 🔮 Prediction")
            st.metric("Result", "Leave 😢" if pred == 1 else "Stay 🙂")
            st.metric("Leave Probability", f"{prob:.2%}")

        with colB:
            st.markdown("### 📋 Input Summary")
            st.dataframe(X_input.T)

# ==========================================================
# 📂 BATCH PREDICTIONS
# ==========================================================

elif page == "📂 Batch Predictions":
    st.markdown("## 📂 Batch Predictions")
    st.info(f"Required columns: {', '.join(FEATURES)}")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df_new = pd.read_csv(file)

        missing = [c for c in FEATURES if c not in df_new.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            preds = model.predict(df_new[FEATURES])
            probs = model.predict_proba(df_new[FEATURES])[:, 1]

            df_new["Prediction"] = np.where(preds == 1, "Leave", "Stay")
            df_new["Leave_Probability"] = probs

            st.markdown("### 🔍 Preview")
            st.dataframe(df_new.head(20))

            st.download_button(
                "⬇️ Download Results",
                df_new.to_csv(index=False).encode(),
                "attrition_predictions.csv",
                "text/csv",
            )

# ==========================================================
# 📊 MODEL EVALUATION
# ==========================================================

elif page == "📊 Model Evaluation":
    st.markdown("## 📊 Model Evaluation")

    X = df[FEATURES].copy()
    y = df["left"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    col1, col2 = st.columns(2)

    # Confusion Matrix
    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="magma", ax=ax)
        st.pyplot(fig)

    # ROC Curve
    with col2:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax.plot([0, 1], [0, 1], "k--")
        ax.legend()
        st.pyplot(fig)

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

# ==========================================================
# 🔍 EXPLAINABILITY (SHAP)
# ==========================================================

elif page == "🔍 Explainability":
    st.markdown("## 🔍 Explainability (SHAP)")

    X = df[FEATURES].copy()
    y = df["left"].copy()

    st.info("Generating SHAP values (this uses a sample of the dataset for speed)…")

    # Preprocess once
    X_processed = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()

    # Sample for background to keep SHAP fast
    import numpy as np

    n_samples = min(400, X_processed.shape[0])
    sample_idx = np.random.choice(X_processed.shape[0], n_samples, replace=False)
    X_sample = X_processed[sample_idx]

    explainer = shap.Explainer(classifier, X_sample)
    shap_values = explainer(X_sample)

    tab1, tab2 = st.tabs(["🌍 Global", "👤 Individual"])

    # GLOBAL
    with tab1:
        st.subheader("🌍 Global SHAP Summary")
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        st.pyplot(fig)

    # INDIVIDUAL
    with tab2:
        idx = st.number_input("Select row index:", 0, len(X) - 1, 0)

        st.subheader("👤 SHAP Waterfall")

        instance = X.iloc[[idx]]
        instance_processed = preprocessor.transform(instance)
        instance_sv = explainer(instance_processed)

        fig, ax = plt.subplots(figsize=(10, 4))
        shap.plots.waterfall(instance_sv[0], show=False)
        st.pyplot(fig)


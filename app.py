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
# SHAP EXPLAINABILITY (FULL FIX)
# ==========================================================

elif page == "🔍 Explainability":
    st.markdown("## 🔍 Explainability (SHAP)")
    st.info("Generating SHAP values (this uses a sample of the dataset for speed)…")

    # Sample the dataset
    X_sample = df[FEATURES].sample(300, random_state=42)

    # Ensure categorical dtype
    for col in categorical_cols:
        if col in X_sample.columns:
            X_sample[col] = X_sample[col].astype("category")

    # 1️⃣ Extract pipeline steps
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    # 2️⃣ Transform input
    X_transformed = preprocessor.transform(X_sample)

    # 3️⃣ Convert sparse → dense (IMPORTANT!)
    if hasattr(X_transformed, "toarray"):
        X_dense = X_transformed.toarray()
    else:
        X_dense = X_transformed

    # 4️⃣ Build SHAP explainer
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_dense)

    # 5️⃣ Normalize SHAP format
    if isinstance(shap_values, list):
        shap_pos = shap_values[1]      # Positive class only
    else:
        if shap_values.ndim == 3:      # (classes, rows, features)
            shap_pos = shap_values[1]
        else:                          # Already correct format
            shap_pos = shap_values

    # Tabs
    tab1, tab2 = st.tabs(["🌍 Global Importance", "👤 Individual Prediction"])

    # ======================================================
    # GLOBAL SUMMARY PLOT
    # ======================================================
    with tab1:
        st.subheader("🌍 Global SHAP Summary Plot")

        fig, ax = plt.subplots(figsize=(10, 5))
        shap.summary_plot(shap_pos, X_dense, show=False)
        st.pyplot(fig)

    # ======================================================
    # INDIVIDUAL WATERFALL
    # ======================================================
    with tab2:
        st.subheader("👤 Individual SHAP Waterfall")

        idx = st.number_input("Select row:", 0, len(X_dense) - 1, 0)

    # Safe base value extraction
    if isinstance(explainer.expected_value, (list, np.ndarray)):
        base_val = explainer.expected_value[1]
    else:
        base_val = explainer.expected_value

    explanation = shap.Explanation(
        values=shap_pos[idx],
        base_values=base_val,
        data=X_dense[idx]
    )

    # Create proper Matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(explanation, max_display=15, show=False)
    st.pyplot(fig)

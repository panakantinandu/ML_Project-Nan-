# =========================
# Employee Attrition ML App - Dark Pro Version
# =========================
import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import gdown

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# =========================
# CONFIG: DATA & MODEL
# =========================

DATA_URL = "https://drive.google.com/uc?id=1oCM6l_7Kx6E9ftLS8C8qjlZ0VWxnUC3Y"  # your Drive link
DATA_PATH = "HR_Data.csv"

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "employee_attrition_pipeline.pkl")


def download_dataset():
    """Download HR_Data.csv from Google Drive if not present."""
    if not os.path.exists(DATA_PATH):
        st.warning("HR_Data.csv not found. Downloading from Google Drive...")
        gdown.download(DATA_URL, DATA_PATH, quiet=False)
        st.success("✅ Dataset downloaded successfully.")


download_dataset()

MODEL_URL = "https://drive.google.com/open?id=13ibNfS8n36ItzzDkJBg0pEMCxEYIDmMd&usp=drive_fs"
MODEL_PATH = "employee_attrition_pipeline.pkl"

def download_model():
    if not os.path.exists(MODEL_PATH):
        st.warning("Downloading ML model…")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        st.success("Model downloaded!")

download_model()



# =========================
# Streamlit Page Config
# =========================
st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark AI-style custom CSS
st.markdown(
    """
    <style>
        .main {
            background-color: #050816;
        }
        .stApp {
            background-color: #050816;
        }
        .stSidebar {
            background-color: #020617 !important;
        }
        .stSidebar, .stSidebar div, .stSidebar section {
            color: #E5E7EB !important;
        }
        h1, h2, h3, h4, h5, h6, p, label, span {
            color: #E5E7EB !important;
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
        .divider {
            border-top: 1px solid #1F2937;
            margin: 1.5rem 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Load Model & Data
# =========================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["left"] = df["Status"].apply(
        lambda x: 0 if str(x).strip().lower() == "active" else 1
    )
    return df


df = load_data()

# Sanity: model must be a Pipeline with preprocessor + classifier
preprocessor = model.named_steps.get("preprocessor", None)
classifier = model.named_steps.get("classifier", None)
if preprocessor is None or classifier is None:
    st.error("❌ Model is not a Pipeline with 'preprocessor' and 'classifier' steps.")
    st.stop()

# =========================
# Sidebar Navigation
# =========================

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/4715/4715321.png",
    width=80,
)
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

# =========================
# 🏠 Home
# =========================

if page == "🏠 Home":
    st.markdown(
        """
        <div class="card">
            <h2 style="text-align:center;">🤖 AI-Powered Employee Retention Insights</h2>
            <p style="text-align:center; color:#9CA3AF;">
                Predict and explain employee attrition using machine learning & model explainability.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Active Employees", int((df["Status"] == "Active").sum()))
    with col2:
        st.metric("Resigned", int((df["Status"] == "Resigned").sum()))
    with col3:
        st.metric("Attrition Rate", f"{df['left'].mean() * 100:.2f}%")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card">
            <h3>✨ What this app does</h3>
            <ul>
                <li>Predicts whether an employee is likely to <b>Stay</b> or <b>Leave</b></li>
                <li>Supports <b>single</b> and <b>batch</b> predictions</li>
                <li>Provides <b>model evaluation</b> (ROC, confusion matrix, metrics)</li>
                <li>Includes <b>global</b> and <b>individual</b> SHAP explainability</li>
            </ul>
            <p style="color:#9CA3AF;">
                Tech stack: <b>Python · Scikit-learn · Streamlit · SHAP</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================
# 🧩 Single Prediction
# =========================

elif page == "🧩 Single Prediction":
    st.markdown("## 🧩 Single Employee Prediction")
    st.write("Fill in the employee details to estimate attrition risk.")

    departments = df["Department"].unique()
    job_titles = df["Job_Title"].unique()
    locations = df["Location"].unique()
    work_modes = df["Work_Mode"].unique()

    with st.form("single_predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            department = st.selectbox("Department", departments)
            location = st.selectbox("Location", locations)
            work_mode = st.selectbox("Work Mode", work_modes)

        with col2:
            job_title = st.selectbox("Job Title", job_titles)
            performance_rating = st.slider("Performance Rating", 1, 5, 3)
            experience_years = st.slider("Years of Experience", 0, 40, 5)

        with col3:
            salary_inr = st.number_input(
                "Annual Salary (INR)",
                min_value=100000,
                max_value=5000000,
                value=800000,
                step=50000,
            )
            submitted = st.form_submit_button("🚀 Predict")

        if submitted:
            input_data = {
                "Department": department,
                "Job_Title": job_title,
                "Location": location,
                "Performance_Rating": performance_rating,
                "Experience_Years": experience_years,
                "Work_Mode": work_mode,
                "Salary_INR": salary_inr,
            }
            X_input = pd.DataFrame([input_data])

            pred = model.predict(X_input)[0]
            prob = model.predict_proba(X_input)[0][1]

            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            colA, colB = st.columns([1, 2])

            with colA:
                st.markdown("### 🔮 Prediction")
                st.metric(
                    "Result",
                    "Leave 😢" if pred == 1 else "Stay 🙂",
                )
                st.metric("Confidence", f"{prob:.2%}")

            with colB:
                st.markdown("### 📋 Input Summary")
                st.dataframe(X_input.T, use_container_width=True)

# =========================
# 📂 Batch Predictions
# =========================

elif page == "📂 Batch Predictions":
    st.markdown("## 📂 Batch Predictions")
    st.write("Upload a CSV file of employees to generate predictions in bulk.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        try:
            new_df = pd.read_csv(uploaded)
            preds = model.predict(new_df)
            new_df["Prediction"] = np.where(preds == 1, "Leave", "Stay")
            st.markdown("### Preview")
            st.dataframe(new_df.head(15), use_container_width=True)

            csv = new_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Predictions CSV",
                csv,
                "batch_predictions.csv",
                "text/csv",
            )
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")
    else:
        st.info("Please upload a CSV with the same feature columns used for training.")

# =========================
# 📊 Model Evaluation
# =========================

elif page == "📊 Model Evaluation":
    st.markdown("## 📊 Model Evaluation & Metrics")

    df["left"] = df["Status"].apply(
        lambda x: 0 if str(x).strip().lower() == "active" else 1
    )
    X = df.drop(
        columns=["Status", "left", "Unnamed: 0", "Employee_ID", "Full_Name", "Hire_Date"]
    )
    y = df["left"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="magma", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    with col2:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color="#6366F1")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig)

    st.subheader("Classification Report")
    report_df = pd.DataFrame(
        classification_report(y_test, y_pred, output_dict=True)
    ).transpose()
    st.dataframe(report_df, use_container_width=True)

# =========================
# 🔍 Explainability (Global + Individual)
# =========================

elif page == "🔍 Explainability":
    st.markdown("## 🔍 Explainability Dashboard")
    st.write(
        "Understand which features drive attrition globally, and why a specific employee gets a prediction."
    )

    tab1, tab2 = st.tabs(["🌍 Global Feature Importance", "👤 Employee-Level Explanation"])

    # ---------- GLOBAL ----------
    with tab1:
        st.markdown("### 🌍 Global Feature Importance (SHAP Summary)")
        try:
            df["left"] = df["Status"].apply(
                lambda x: 0 if str(x).strip().lower() == "active" else 1
            )
            X = df.drop(
                columns=[
                    "Status",
                    "left",
                    "Unnamed: 0",
                    "Employee_ID",
                    "Full_Name",
                    "Hire_Date",
                ]
            )
            y = df["left"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Preprocess and sample
            X_processed = preprocessor.transform(X_test)
            feature_names = preprocessor.get_feature_names_out()
            X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

            X_sample = X_processed_df.sample(
                min(300, len(X_processed_df)), random_state=42
            )

            explainer = shap.Explainer(classifier, X_sample)
            shap_values = explainer(X_sample)

            fig, ax = plt.subplots(figsize=(9, 5))
            shap.summary_plot(shap_values, X_sample, show=False)
            st.pyplot(fig)

            st.markdown(
                """
                **How to read this plot:**
                - Each dot = one employee  
                - X-axis = impact on prediction (more right = more likely to leave)  
                - Color = feature value (red = high, blue = low)  
                """
            )
        except Exception as e:
            st.error(f"Global SHAP failed: {e}")

    # ---------- INDIVIDUAL ----------
    with tab2:
        st.markdown("### 👤 Individual Employee SHAP Waterfall")
        st.write("Pick an employee from the test set to see a breakdown of their prediction.")

        try:
            df["left"] = df["Status"].apply(
                lambda x: 0 if str(x).strip().lower() == "active" else 1
            )
            X = df.drop(
                columns=[
                    "Status",
                    "left",
                    "Unnamed: 0",
                    "Employee_ID",
                    "Full_Name",
                    "Hire_Date",
                ]
            )
            y = df["left"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Reset index to keep mapping back to original df
            X_test_reset = X_test.reset_index().rename(columns={"index": "orig_index"})

            X_processed = preprocessor.transform(X_test)
            feature_names = preprocessor.get_feature_names_out()
            X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

            # Build dropdown options
            employee_choices = []
            for row_idx, row in X_test_reset.iterrows():
                orig_idx = row["orig_index"]
                full_name = df.loc[orig_idx, "Full_Name"]
                dept = df.loc[orig_idx, "Department"]
                employee_choices.append(f"{row_idx} — {full_name} ({dept})")

            selected = st.selectbox("Select Employee", employee_choices)
            selected_row_idx = int(selected.split(" — ")[0])

            raw_input = X_test_reset.iloc[[selected_row_idx]].drop(
                columns=["orig_index"]
            )
            processed_input = X_processed_df.iloc[[selected_row_idx]]

            pred = model.predict(raw_input)[0]
            prob = model.predict_proba(raw_input)[0][1]

            colA, colB = st.columns([1, 2])
            with colA:
                st.metric("Prediction", "Leave 😢" if pred == 1 else "Stay 🙂")
                st.metric("Confidence", f"{prob:.2%}")
            with colB:
                st.markdown("#### Employee Feature Snapshot")
                st.dataframe(raw_input.T, use_container_width=True)

            shap.initjs()
            # Use a small background sample for explainer
            background = X_processed_df.sample(
                min(300, len(X_processed_df)), random_state=42
            )
            explainer = shap.Explainer(classifier, background)
            shap_values = explainer(processed_input)

            st.markdown("#### 📊 SHAP Waterfall Explanation")
            fig, ax = plt.subplots(figsize=(10, 5))
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Individual SHAP failed: {e}")

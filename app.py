
# =========================
# Employee Attrition ML App (Final Polished Version)
# =========================
import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc


# =========================
# Streamlit Page Config
# =========================
st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS for better visuals
st.markdown("""
    <style>
        .main {
            background-color: #F9FAFB;
        }
        .stButton button {
            background-color: #6C5CE7;
            color: white;
            font-weight: 600;
            border-radius: 8px;
            padding: 0.6em 1.2em;
        }
        .stButton button:hover {
            background-color: #5B4DE0;
            transform: scale(1.02);
        }
        .stSidebar {
            background-color: #FFFFFF;
        }
        h1, h2, h3 {
            color: #2C3E50;
        }
        .metric-container {
            text-align: center;
            background-color: #FFFFFF;
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)


# =========================
# Load Model & Data
# =========================
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "employee_attrition_pipeline.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

@st.cache_data
def load_data():
    df = pd.read_csv("HR_Data.csv")
    df["left"] = df["Status"].apply(lambda x: 0 if str(x).strip().lower() == "active" else 1)
    return df

df = load_data()


# =========================
# Sidebar Navigation
# =========================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2965/2965879.png", width=80)
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Choose a section:",
    ["🏠 Home", "🧩 Single Prediction", "📂 Batch Predictions",
     "📊 Model Evaluation", "🔍 Global Explainability", "👤 Employee Explainability"]
)



# =========================
# 🏠 Home
# =========================
if page == "🏠 Home":
    st.markdown("""
    <div style="background-color:#6C5CE7;padding:10px;border-radius:10px;margin-bottom:20px;">
        <h2 style="color:white;text-align:center;">AI-Powered Employee Retention Insights</h2>
    </div>
    """, unsafe_allow_html=True)

    st.title("🏢 Employee Attrition Prediction System")
    st.write("""
    This application predicts whether an employee is likely to **stay** or **leave** the company
    based on HR analytics and historical data.
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Active Employees", df[df["Status"] == "Active"].shape[0])
    col2.metric("Resigned", df[df["Status"] == "Resigned"].shape[0])
    col3.metric("Attrition Rate", f"{df['left'].mean() * 100:.2f}%")

    st.markdown("""
    ---
    **Features**
    - 🔹 Predict individual employee attrition  
    - 🔹 Run batch predictions from uploaded CSVs  
    - 🔹 View model performance & key insights  
    - 🔹 Understand predictions with SHAP explainability  

    **Tech Stack:** Python • Scikit-learn • Streamlit • SHAP
    """)


# =========================
# 🧩 Single Prediction
# =========================
elif page == "🧩 Single Prediction":
    st.markdown("### 🧩 Predict Single Employee Attrition")
    st.write("Provide employee details to predict whether they are likely to stay or leave.")

    departments = df["Department"].unique()
    job_titles = df["Job_Title"].unique()
    locations = df["Location"].unique()
    work_modes = df["Work_Mode"].unique()

    with st.form("prediction_form"):
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
            salary_inr = st.number_input("Annual Salary (INR)", min_value=100000, max_value=5000000, value=800000)
            submitted = st.form_submit_button("🚀 Predict")

        if submitted:
            input_data = {
                "Department": department,
                "Job_Title": job_title,
                "Location": location,
                "Performance_Rating": performance_rating,
                "Experience_Years": experience_years,
                "Work_Mode": work_mode,
                "Salary_INR": salary_inr
            }
            X_input = pd.DataFrame([input_data])
            prediction = model.predict(X_input)[0]
            prob = model.predict_proba(X_input)[0][1]

            st.markdown("### 🧠 Prediction Result")
            st.success(f"Prediction: {'Employee will Leave 😢' if prediction == 1 else 'Employee will Stay 🙂'}")
            st.info(f"Confidence: {prob:.2%}")


# =========================
# 📂 Batch Predictions
# =========================
elif page == "📂 Batch Predictions":
    st.markdown("### 📂 Batch Predictions")
    st.write("Upload a CSV file to predict attrition for multiple employees at once.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        new_df = pd.read_csv(uploaded)
        preds = model.predict(new_df)
        new_df["Prediction"] = np.where(preds == 1, "Leave", "Stay")

        st.dataframe(new_df.head(10), use_container_width=True)

        csv = new_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Predictions", csv, "batch_predictions.csv", "text/csv")


# =========================
# 📊 Model Evaluation
# =========================
elif page == "📊 Model Evaluation":
    st.markdown("### 📊 Model Evaluation & Metrics")

    df["left"] = df["Status"].apply(lambda x: 0 if str(x).strip().lower() == "active" else 1)
    X = df.drop(columns=["Status", "left", "Unnamed: 0", "Employee_ID", "Full_Name", "Hire_Date"])
    y = df["left"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color="#6C5CE7")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.legend()
        st.pyplot(fig)

    st.subheader("Classification Report")
    st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())

# =========================
# 🔍 Unified Explainability Dashboard
# =========================
elif page == "🔍 Explainability":
    st.markdown("## 🔍 Explainability Dashboard")
    st.write("""
    This section provides two complementary explainability views:
    - **Global Explainability:** which features drive attrition overall  
    - **Employee-Level Explainability:** why a specific employee is predicted to stay or leave
    """)

    # Tabs for global vs individual analysis
    tab1, tab2 = st.tabs(["🌍 Global Feature Importance", "👤 Individual Employee Analysis"])

    # ========== 🌍 GLOBAL TAB ==========
    with tab1:
        st.markdown("### 🌍 Global Feature Importance")
        st.caption("Understanding overall model behavior across all employees.")

        try:
            df["left"] = df["Status"].apply(lambda x: 0 if str(x).strip().lower() == "active" else 1)
            X = df.drop(columns=["Status", "left", "Unnamed: 0", "Employee_ID", "Full_Name", "Hire_Date"])
            y = df["left"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            preprocessor = model.named_steps["preprocessor"]
            classifier = model.named_steps["classifier"]

            X_processed = preprocessor.transform(X_test)
            feature_names = preprocessor.get_feature_names_out()
            X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

            X_sample = X_processed_df.sample(min(300, len(X_processed_df)), random_state=42)

            explainer = shap.Explainer(classifier, X_sample)
            shap_values = explainer(X_sample)

            col1, col2 = st.columns([3, 2])
            with col1:
                st.subheader("Top Feature Impact Summary")
                fig, ax = plt.subplots(figsize=(9, 5))
                shap.summary_plot(shap_values, X_sample, show=False)
                st.pyplot(fig)
            with col2:
                st.markdown("""
                **Insights:**
                - 🔹 Features with wider spread have higher global influence  
                - 🔹 Colors represent correlation with attrition (red → leave, blue → stay)  
                - 🔹 Salary, performance, and experience usually dominate attrition signals
                """)

        except Exception as e:
            st.error(f"Global SHAP failed: {e}")


    # ========== 👤 INDIVIDUAL TAB ==========
    with tab2:
        st.markdown("### 👤 Individual Employee Explainability (SHAP Waterfall)")
        st.caption("Select an employee to visualize personalized prediction reasoning.")

        try:
            df["left"] = df["Status"].apply(lambda x: 0 if str(x).strip().lower() == "active" else 1)
            X = df.drop(columns=["Status", "left", "Unnamed: 0", "Employee_ID", "Full_Name", "Hire_Date"])
            y = df["left"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            preprocessor = model.named_steps["preprocessor"]
            classifier = model.named_steps["classifier"]

            X_processed = preprocessor.transform(X_test)
            feature_names = preprocessor.get_feature_names_out()
            X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

            # Build employee dropdown list
            sample_indices = X_test.index
            employee_choices = [
                f"{i} — {df.loc[i, 'Full_Name']} ({df.loc[i, 'Department']})"
                for i in sample_indices
            ]
            selected = st.selectbox("Select Employee", employee_choices)

            # Extract selection
            selected_idx = int(selected.split(" — ")[0])
            raw_input = X_test.loc[[selected_idx]]
            processed_input = X_processed_df.loc[[list(sample_indices).index(selected_idx)]]

            # Model prediction
            pred = model.predict(raw_input)[0]
            prob = model.predict_proba(raw_input)[0][1]

            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Prediction", "Leave 😢" if pred == 1 else "Stay 🙂")
                st.metric("Confidence", f"{prob:.2%}")
            with col2:
                st.dataframe(raw_input.T, use_container_width=True)

            # SHAP explanation for that employee
            shap.initjs()
            explainer = shap.Explainer(classifier, X_processed_df.sample(300, random_state=42))
            shap_values = explainer(processed_input)

            st.subheader("📊 SHAP Waterfall Explanation")
            fig, ax = plt.subplots(figsize=(10, 5))
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Individual SHAP failed: {e}")

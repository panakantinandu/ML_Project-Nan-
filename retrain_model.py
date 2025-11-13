# =========================
# Model Retraining Script
# =========================
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# 1️⃣ Load Data
df = pd.read_csv("HR_Data.csv")
df["left"] = df["Status"].apply(lambda x: 0 if str(x).strip().lower() == "active" else 1)

# 2️⃣ Split Features / Target
X = df.drop(columns=["Status", "left", "Unnamed: 0", "Employee_ID", "Full_Name", "Hire_Date"])
y = df["left"]

# 3️⃣ Define Columns by Type
numeric_features = ["Performance_Rating", "Experience_Years", "Salary_INR"]
low_card_features = ["Department", "Work_Mode"]
high_card_features = ["Job_Title", "Location"]

# 4️⃣ Build Transformers
numeric_transformer = StandardScaler()
low_card_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
high_card_transformer = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("low", low_card_transformer, low_card_features),
        ("high", high_card_transformer, high_card_features)
    ]
)

# 5️⃣ Pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=500))
])

# 6️⃣ Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 7️⃣ Train and Save
model.fit(X_train, y_train)
joblib.dump(model, "employee_attrition_pipeline.pkl")

print("✅ Model retrained and saved successfully.")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# ==================================================
# 1. LOAD DATA
# ==================================================
print("Loading dataset...")
df = pd.read_csv("HR_Data.csv")

# ==================================================
# 2. OPTIONAL: Sample large dataset for faster training
# ==================================================
MAX_ROWS = 20000   # Safe for any laptop

if df.shape[0] > MAX_ROWS:
    df = df.sample(n=MAX_ROWS, random_state=42)
    print(f"Dataset too large. Using sample of {MAX_ROWS} rows.")
else:
    print(f"Dataset size OK: {df.shape[0]} rows")

print(df.head())
print(df.columns)

# ==================================================
# 3. PREPARE TARGET
# ==================================================
target = "Status"    # Your actual target column

if target not in df.columns:
    raise ValueError(f"❌ ERROR: Target column '{target}' not found in dataset!")

y = df[target]
X = df.drop(columns=[target])

# ==================================================
# 4. IDENTIFY COLUMN TYPES
# ==================================================
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

print("Numeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

# ==================================================
# 5. PREPROCESSORS
# ==================================================
numeric_transformer = StandardScaler()

categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# ==================================================
# 6. MODEL
# ==================================================
model = LogisticRegression(
    max_iter=1000,
    n_jobs=-1
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# ==================================================
# 7. TRAIN / TEST SPLIT
# ==================================================
print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==================================================
# 8. TRAIN MODEL
# ==================================================
print("Training model...")
pipeline.fit(X_train, y_train)

# ==================================================
# 9. EVALUATE
# ==================================================
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n🎉 Model trained successfully!")
print(f"Accuracy: {acc:.4f}")

# ==================================================
# 10. SAVE MODEL
# ==================================================
MODEL_PATH = "employee_attrition_pipeline.pkl"
joblib.dump(pipeline, MODEL_PATH)

print(f"\n💾 Model saved as: {MODEL_PATH}")
print("Done!")

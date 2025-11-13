# 🏢 Employee Attrition Prediction System

### AI-Powered Workforce Analytics | Streamlit • Scikit-Learn • SHAP • Python

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-yellow?style=for-the-badge">
  <img src="https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge">
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/Explainability-SHAP-purple?style=for-the-badge">
  <img src="https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=for-the-badge">
</p>

A complete end-to-end Machine Learning application built to predict **employee attrition** and provide **explainable insights** using SHAP.
This project delivers a real-world HR analytics experience with a **modern dark UI**, **batch predictions**, and a **smart explainability dashboard**.

---

## 🚀 Features

### 🔮 Prediction Capabilities

* **Single employee prediction**
* **Batch predictions** from CSV uploads
* **Prediction confidence scores**

### 📊 Analytics & Evaluation

* Confusion Matrix
* ROC Curve
* Classification Report
* Attrition Statistics Dashboard

### 🔍 Explainability (SHAP)

* **Global Feature Importance**
* **Individual Employee Waterfall Plots**
* **Interactive Explainability Dashboard**

### 🎨 UI & Product Experience

* Fully custom **dark theme**
* Modern CSS styling
* Sidebar navigation
* Metric cards & clean layout

### 🗂️ Deployment Ready

* Google Drive dataset auto-downloader
* Modular codebase
* Works instantly on any machine

---

## 🏗️ Project Structure

```
ML_PROJECT/
│── app.py                     # Main Streamlit App
│── employee_attrition_pipeline.pkl   # Trained ML Model
│── retrain_model.py           # Script to retrain the model
│── requirements.txt           # Dependencies
│── README.md                 # Documentation
│── .gitignore
│── .streamlit/
│     └── config.toml          # Dark Theme Config
```

---

## 📦 Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the app

```bash
streamlit run app.py
```

### 4️⃣ Dataset

The dataset **downloads automatically** from Google Drive on first run.

No manual action needed.

---

## 🧠 Model Details

### Algorithms Used

* Random Forest (or your model)
* OneHotEncoding + StandardScaler (pipeline)

### Target

`Status → Active (0) / Left (1)`

### Core Input Features:

* Department
* Job Title
* Performance Rating
* Experience
* Salary
* Work Mode
* Location

---

## 📊 Screenshots

*Add screenshots after pushing to GitHub (Streamlit UI, SHAP charts, etc.).*

Example layout sections you should upload:

* Home Dashboard
* Single Prediction
* Batch Upload
* Model Evaluation
* SHAP Global Importance
* SHAP Waterfall

---

## 📁 Requirements

Everything is already in `requirements.txt`, including:

* streamlit
* scikit-learn
* shap
* pandas
* numpy
* seaborn
* matplotlib
* joblib
* gdown

---

## 🤝 Contributing

Pull requests are welcome.
For major changes, open an issue first to discuss what you’d like to modify.

---

## 📄 License

This project is **open-source** and free to use.

---

## ⭐ Show Your Support

If this project helped you, **give the repo a star** on GitHub!

---


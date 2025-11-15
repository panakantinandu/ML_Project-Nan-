# 🏢 Employee Attrition Prediction System 
### AI-Powered HR Analytics • LightGBM • Streamlit • SHAP • Python

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-yellow?style=for-the-badge">
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge">
  <img src="https://img.shields.io/badge/Machine%20Learning-LightGBM-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/Explainability-SHAP-purple?style=for-the-badge">
  <img src="https://img.shields.io/badge/Deployment-Render-brightgreen?style=for-the-badge">
</p>

A production-ready **Employee Attrition Prediction System** that uses **LightGBM**, a highly efficient gradient boosting algorithm, combined with **OneHotEncoding pipelines**, **explainability (SHAP)**, and a polished **modern UI built with Streamlit**.

Designed for **real-world HR analytics**, this application performs:

✔ Real-time predictions
✔ Batch scoring
✔ Full evaluation metrics
✔ Explainable ML insights
✔ Ready for cloud deployment

---

# 🚀 Features

### 🔮 Prediction System

* Single employee attrition prediction
* Batch predictions via CSV file
* Leave/Stay probability scores
* Intelligent preprocessing pipeline built into the model

---

### 📊 Analytics Dashboard

* Attrition statistics
* Department-wise breakdown
* Confusion Matrix
* ROC-AUC Curve
* Classification report

---

### 🔍 Explainability (SHAP)

* Global feature importance
* Individual prediction waterfall plots
* Helps HR understand *why* a prediction happens

---

### 🎨 Modern UI

* Fully customized dark theme
* Gradient buttons & card layout
* Sidebar navigation
* Smooth user experience

---

# 🏗️ Project Structure

```
ML_PROJECT/
│── app.py                             # Main Streamlit application
│── employee_attrition_pipeline.pkl    # Trained ML pipeline (LightGBM + OHE)
│── Realistic_HR_Attrition_3000.csv    # Realistic training/demo dataset
│── retrain_model.py                   # Model training script (Pipeline)
│── requirements.txt                   # Dependencies for deployment
│── README.md                          # Documentation (this file)
│── .gitignore
│── .streamlit/
│     └── config.toml                  # Dark theme configuration
```

---

# 📦 Installation & Setup

## 1️⃣ Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

## 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

## 3️⃣ Run locally

```bash
streamlit run app.py
```

---

# 🔥 Training the Model (LightGBM Pipeline)

The entire ML workflow is inside:

```
retrain_model.py
```

It uses:

* LightGBM classifier
* OneHotEncoder inside a ColumnTransformer
* Full preprocessing + model stored inside one pipeline
* Saved as:

  ```
  employee_attrition_pipeline.pkl
  ```

The dataset used:

```
Realistic_HR_Attrition_3000.csv
```

This dataset is **synthetic but highly realistic**, built with actual HR attrition patterns.

---

# 🌐 Deployment (Render – Easiest & Free)

### 1️⃣ Push project to GitHub

Make sure your repo contains:

```
app.py  
requirements.txt  
employee_attrition_pipeline.pkl  
Realistic_HR_Attrition_3000.csv
```

### 2️⃣ Go to Render.com → New → Web Service

Choose your GitHub Repo.

### 3️⃣ Configure Render

**Build Command**

```
pip install -r requirements.txt
```

**Start Command**

```
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

**Instance Type**

```
Free Tier
```

### 4️⃣ Deploy

Render will give you a public link like:

```
https://your-project.onrender.com
```

You’re live! 🚀

---

# 📁 Requirements

Your `requirements.txt` should include:

```
streamlit
pandas
numpy
lightgbm
scikit-learn
matplotlib
seaborn
shap
joblib
```

(Optional but recommended)

```
pyyaml
plotly
```

---

# 📊 Screenshots


### 🏠 Home Dashboard
<img width="1903" height="895" alt="image" src="https://github.com/user-attachments/assets/60e621fa-3b9f-4901-b55e-56d1a97afdd2" />


### 🔮 Single Prediction
<img width="1546" height="843" alt="image" src="https://github.com/user-attachments/assets/b7967ffe-cd96-4e04-b52e-b90d0763ceb6" />


### 📂 Batch Prediction
<img width="1558" height="711" alt="image" src="https://github.com/user-attachments/assets/8bae5f33-da21-409b-9ca4-2babf8e751a7" />


### 📊 Evaluation Dashboard
<img width="1599" height="892" alt="image" src="https://github.com/user-attachments/assets/578be7ab-ba74-4ce8-9420-2c427234a88b" />


### 🔍 SHAP Explainability

---

# 🤝 Contributing

Pull requests are welcome.
For major changes, open an issue to discuss your ideas.

---

# ⭐ Support

If this project helped you, **please star the repo** on GitHub.
It motivates further improvements.

---

# 🔒 License

This project is **100% open-source**.

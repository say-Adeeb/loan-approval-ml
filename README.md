# 💰 Loan Approval Prediction System

Machine Learning powered web application that predicts loan approval status using **XGBoost** and is deployed live using **Streamlit Cloud**.

---

## 🚀 Live Demo

🔗 **Live App:**  
https://say-adeeb-loan-approval-ml-srcapp-imnvsk.streamlit.app/
---

## 📌 Project Overview

This project is an end-to-end Machine Learning system that:

- Trains multiple classification models  
- Selects the best-performing model (XGBoost)  
- Saves the trained model using `joblib`  
- Deploys the model using Streamlit  
- Displays prediction confidence  
- Visualizes top feature importance  
- Supports dark mode UI  

---

## 🧠 Problem Statement

Financial institutions need a reliable way to determine whether a loan applicant should be approved based on:

- Demographics  
- Financial status  
- Credit history  
- Loan intent  

This application predicts loan approval based on applicant details.

---

## ⚙️ Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- XGBoost  
- Streamlit  
- Matplotlib  
- Joblib  

---

## 🏗 Project Structure

```
loan-approval-ml/
│
├── data/
├── models/
├── src/
│   ├── __init__.py
│   ├── app.py
│   ├── predict.py
│   └── train.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🔍 Features

-> Predict loan approval (Approved / Rejected)  
-> Display prediction confidence (%)  
-> Progress bar visualization  
-> Top 10 Feature Importance chart  
-> Dark mode toggle  
-> Clean and professional UI  
-> Fully deployed on Streamlit Cloud  

---

## 📊 Model Training

Models trained and evaluated:

- Decision Tree  
- Random Forest  
- Logistic Regression  
- XGBoost (Selected Best Model)  

Evaluation Metrics Used:

- Accuracy  
- Precision  
- Recall  
- F1 Score  

---

## How to Run Locally

### 1️. Clone the repository

```bash
git clone https://github.com/say-Adeeb/loan-approval-ml.git
cd loan-approval-ml
```

### 2️. Install dependencies

```bash
pip install -r requirements.txt
```

### 3️. Run the app

```bash
streamlit run src/app.py
```

---

## 🌍 Deployment

This project is deployed using **Streamlit Cloud**.

Steps:
1. Push project to GitHub  
2. Connect repo to Streamlit Cloud  
3. Deploy using `src/app.py`  

---

## 👨‍💻 Author

**Syed Adeeb**

Built with focus on real-world ML deployment and production-ready structure.

---
# Loan Approval Prediction

Machine Learning model to predict loan approval status.

## Tech Stack
- Python
- Pandas
- Scikit-learn
- XGBoost

💰 Loan Approval Prediction System

Machine Learning powered web application that predicts loan approval status using XGBoost and is deployed live using Streamlit Cloud.

🚀 Live Demo

👉 Live App:
https://say-adeeb-loan-approval-ml-srcapp-imnvsk.streamlit.app/


📌 Project Overview

This project is an end-to-end Machine Learning system that:

Trains multiple classification models

Selects the best-performing model (XGBoost)

Saves the trained model using joblib

Deploys the model using Streamlit

Displays prediction confidence

Visualizes top feature importance

Supports dark mode UI

🧠 Problem Statement

Financial institutions need a reliable way to determine whether a loan applicant should be approved based on:

Demographics

Financial status

Credit history

Loan intent

This application predicts loan approval based on applicant details.

⚙️ Tech Stack

Python

Pandas

NumPy

Scikit-learn

XGBoost

Streamlit

Matplotlib

Joblib

🏗 Project Structure
loan-approval-ml/
│
├── data/
│   └── loan_data.csv
│
├── models/
│   ├── model.pkl
│   └── columns.pkl
│
├── src/
│   ├── app.py
│   ├── predict.py
│   └── train.py
│
├── requirements.txt
└── README.md
🔍 Features

-> Predict loan approval (Approved / Rejected)
-> Display prediction confidence (%)
-> Progress bar visualization
-> Top 10 Feature Importance chart
-> Clean and professional UI
-> Fully deployed on Streamlit Cloud

📊 Model Training

Models trained and evaluated:

Decision Tree

Random Forest

Logistic Regression

XGBoost (Selected Best Model)

Evaluation Metrics Used:

Accuracy

Precision

Recall

F1 Score

XGBoost was selected based on overall performance.

📈 Feature Importance

The app dynamically displays the top 10 most important features influencing loan approval decisions.

This enhances model interpretability and transparency.

🖥 How to Run Locally
1️⃣ Clone the repository
git clone https://github.com/yourusername/loan-approval-ml.git
cd loan-approval-ml
2️⃣ Install dependencies
pip install -r requirements.txt
3️⃣ Run the app
streamlit run src/app.py
🌍 Deployment

This project is deployed using:

Streamlit Cloud

Deployment steps:

Push project to GitHub

Connect repo to Streamlit Cloud

Deploy using src/app.py

🎯 Key Learnings

Model serialization using joblib

Handling feature alignment in deployment

Managing relative paths in cloud deployment

Creating interactive ML dashboards

Git & GitHub version control

Production-ready project structuring

👨‍💻 Author

Syed Adeeb

Built with focus on real-world ML deployment and production-ready structure.
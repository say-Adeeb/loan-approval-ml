
#DEPLOYMENT USING STREAMLIT WEB APPLICATION - LOAN PREDICTION SYSTEM

#streamlit for UI creation
#custom prediction function from src module

#web-based inference interface built with Streamlit
#enables real-time loan approval prediction using trained ML models
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from predict import predict, model, columns

#PAGE CONFIG
st.set_page_config(
    page_title="Loan Approval System",
    page_icon="💰",
    layout="centered"
)

#DARK MODE TOGGLE
dark_mode = st.toggle("🌙 Dark Mode")

if dark_mode:
    st.markdown(
        """
        <style>
        body {
            background-color: #0E1117;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

#HEADER
st.markdown("""
    <h1 style='text-align: center; color: #2E86C1;'>
    💰 Loan Approval Prediction System
    </h1>
""", unsafe_allow_html=True)

st.markdown("""
<p style='text-align: center;'>
Machine Learning based Loan Eligibility Prediction using XGBoost
</p>
""", unsafe_allow_html=True)

st.divider()

#INPUT SECTION
st.subheader("📋 Applicant Information")

col1, col2 = st.columns(2)

with col1:
    person_age = st.number_input("Age", min_value=18, max_value=100, step=1)
    person_income = st.number_input("Annual Income", min_value=0)
    person_emp_exp = st.number_input("Employment Experience (Years)", min_value=0, step=1)
    loan_amnt = st.number_input("Loan Amount Requested", min_value=0)

with col2:
    loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0)
    loan_percent_income = st.number_input("Loan Percent of Income", min_value=0.0)
    cb_person_cred_hist_length = st.number_input("Credit History Length (Years)", min_value=0, step=1)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, step=1)

st.divider()

st.subheader("🏠 Personal & Loan Details")

col3, col4 = st.columns(2)

with col3:
    person_gender = st.selectbox("Gender", ["Male", "Female"])
    person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE"])

with col4:
    loan_intent = st.selectbox(
        "Loan Purpose",
        ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
    )
    person_education = st.selectbox(
        "Education Level",
        ["Doctorate", "Master", "Bachelor", "High School", "Associate"]
    )
    previous_loan_defaults_on_file = st.selectbox(
        "Previous Loan Default",
        ["Yes", "No"]
    )

st.divider()

#PREDICTION
if st.button("🔍 Predict Loan Approval"):

    input_data = {
        "person_age": person_age,
        "person_income": person_income,
        "person_emp_exp": person_emp_exp,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "credit_score": credit_score,
        "person_gender": person_gender,
        "person_home_ownership": person_home_ownership,
        "loan_intent": loan_intent,
        "person_education": person_education,
        "previous_loan_defaults_on_file": previous_loan_defaults_on_file
    }

    result, prob = predict(input_data)

    confidence = round(prob * 100, 2)

    st.divider()

    if result == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.progress(int(confidence))
    st.write(f"Confidence: **{confidence}%**")

    #FEATURE IMPORTANCE
    st.subheader("📊 Top 10 Feature Importance")

    importances = model.feature_importances_
    feature_names = columns

    importance_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values(by="Importance", ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots()
    ax.barh(importance_df["Feature"], importance_df["Importance"])
    ax.invert_yaxis()
    st.pyplot(fig)

#FOOTER
st.divider()
st.caption("Built using XGBoost & Streamlit | Developed by Syed Adeeb")
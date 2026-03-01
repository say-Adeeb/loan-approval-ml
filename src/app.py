
#DEPLOYMENT USING STREAMLIT WEB APPLICATION - LOAN PREDICTION SYSTEM

#streamlit for UI creation
#custom prediction function from src module

#web-based inference interface built with Streamlit
#enables real-time loan approval prediction using trained ML model

import streamlit as st
from predict import predict

st.title("Loan Prediction App")

st.write("Enter applicant details below:")

person_age = st.number_input("Age")
person_income = st.number_input("Income")
person_emp_exp = st.number_input("Employment Experience")
loan_amnt = st.number_input("Loan Amount")
loan_int_rate = st.number_input("Interest Rate")

loan_percent_income = st.number_input("Loan Percent Income")
cb_person_cred_hist_length = st.number_input("Credit History Length")
credit_score = st.number_input("Credit Score")

person_gender = st.selectbox("Gender", ["Male", "Female"])
person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE"])
loan_intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
person_education = st.selectbox("Education", ["Doctorate", "Master", "Bachelor", "High School", "Associate"])
previous_loan_defaults_on_file = st.selectbox("Previous Loan Default", ["Yes", "No"])

if st.button("Predict"):

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

    result = predict(input_data)

    if result == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")
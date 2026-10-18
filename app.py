import streamlit as st
import pandas as pd
import joblib

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('columns.pkl')
num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

st.title("Loanlytics")
st.write("Loan approval prediction using Logistic Regression")

c1, c2 = st.columns(2)
with c1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", [0, 1, 2, 3])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_emp = st.selectbox("Self Employed", ["No", "Yes"])
    area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
with c2:
    app_income = st.number_input("Applicant Income", min_value=0, value=5000)
    co_income = st.number_input("Coapplicant Income", min_value=0, value=0)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, value=130)
    term = st.selectbox("Loan Term (months)", [360, 180, 480, 300, 240, 120, 84, 60, 36, 12])
    credit = st.selectbox("Credit History (1 = good, 0 = bad)", [1, 0])

if st.button("Predict"):
    row = {
        'Gender': 1 if gender == 'Male' else 0,
        'Married': 1 if married == 'Yes' else 0,
        'Dependents': dependents,
        'Education': 1 if education == 'Graduate' else 0,
        'Self_Employed': 1 if self_emp == 'Yes' else 0,
        'ApplicantIncome': app_income,
        'CoapplicantIncome': co_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': term,
        'Credit_History': credit,
        'Property_Area_Semiurban': 1 if area == 'Semiurban' else 0,
        'Property_Area_Urban': 1 if area == 'Urban' else 0
    }
    data = pd.DataFrame([row])[columns].astype(float)
    data[num_cols] = scaler.transform(data[num_cols])
    prob = model.predict_proba(data)[0][1]

    st.write(f"Approval probability: {prob*100:.1f}%")
    if prob >= 0.5:
        st.success("Prediction: APPROVED")
    else:
        st.error("Prediction: REJECTED")
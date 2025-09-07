# Import
import pandas as pd
import streamlit as st
import joblib
import os

# Load pipeline
model_path = "churn_pipeline.pkl"
if os.path.exists(model_path):
    pipeline = joblib.load(model_path)
    st.success("Model load successfully")
else:
    st.warning(f'Model file {model_path} dose not exists')

# App Heading
st.title("Customer Churn Predictiction")
st.write("Fill your details")
# Form Inputs
with st.form("Churn_Form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("SeniorCitizen", ["Yes", "No"])
    Partner = st.selectbox("Partner", ['Yes', 'No'])
    Dependents = st.selectbox("Dependents", ['Yes', 'No'])
    tenure = st.number_input("Tenure (In Months)", min_value=0)
    PhoneService = st.selectbox("PhoneService", ['Yes', 'No'])
    MultipleLines = st.selectbox("MultipleLines", ['No phone service' 'No' 'Yes'])
    InternetService = st.selectbox("InternetService", ['DSL' 'Fiber optic' 'No'])
    OnlineSecurity = st.selectbox("OnlineSecurity", ['No' 'Yes' 'No internet service'])
    OnlineBackup = st.selectbox("OnlineBackup", ['Yes' 'No' 'No internet service'])
    DeviceProtection = st.selectbox("DeviceProtection", ['No' 'Yes' 'No internet service'])
    TechSupport = st.selectbox("TechSupport", ['No' 'Yes' 'No internet service'])
    StreamingTV = st.selectbox("StreamingTV", ['No' 'Yes' 'No internet service'])
    StreamingMovies = st.selectbox("StreamingMovies", ['No' 'Yes' 'No internet service'])
    Contract = st.selectbox("Contract", ['Month-to-month' 'One year' 'Two year'])
    PaperlessBilling = st.selectbox("PaperlessBilling", ['Yes' 'No'])
    PaymentMethod = st.selectbox("PaymentMethod", ['Electronic check' 'Mailed check' 'Bank transfer (automatic)'
 'Credit card (automatic)'])
    MonthlyCharges = st.number_input("MonthlyCharges", min_value=0.0)
    TotalCharges = st.number_input("TotalCharges", min_value=0.0)

    submitted = st.form_submit_button("Predict")

# Create dataframe
if submitted:
    input_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
        }])
    
    prediction = pipeline.predict(input_data)[0]
    probability = pipeline.predict_proba(input_data)[0][1]
    st.success(f"Prediction: {'Yes' if prediction == 1 else 'No'}")
    st.write(f"Churn Probability:  {probability*100:.2f}%") 


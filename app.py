# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model
with open("model.pkl", "rb") as file:
    model, le = pickle.load(file)

st.set_page_config(page_title="Car Purchase Prediction App", layout="wide")

# Main Title
st.title("🚗 Car Purchase Amount Prediction")
st.write("Use this app to predict the car purchase amount based on user financial details.")

# Sidebar inputs
st.sidebar.header("🔧 Input Features")

gender = st.sidebar.selectbox("Gender", le.classes_)
age = st.sidebar.slider("Age", 18, 70, 30)
annual_salary = st.sidebar.number_input("Annual Salary ($)", min_value=10000, max_value=200000, value=50000, step=1000)
credit_card_debt = st.sidebar.number_input("Credit Card Debt ($)", min_value=0, max_value=50000, value=5000, step=500)
net_worth = st.sidebar.number_input("Net Worth ($)", min_value=0, max_value=1000000, value=100000, step=5000)

# Predict button
if st.sidebar.button("💰 Predict Car Purchase Amount"):
    gender_encoded = le.transform([gender])[0]
    input_data = np.array([[gender_encoded, age, annual_salary, credit_card_debt, net_worth]])
    prediction = model.predict(input_data)[0]

    st.subheader("Predicted Car Purchase Amount")
    st.success(f"Estimated Amount: **${prediction:,.2f}**")

# Dataset preview
st.markdown("---")
st.subheader("📊 Dataset Preview")
data = pd.read_csv("D:\Darshil\Study_material\Red & White\Supervised Learning Algorithms\Exams\EXAM-5\Car Purchasing\Car_Purchasing_Data.csv")
st.dataframe(data.head())

# Footer
st.markdown("---")
st.caption("Developed by Darshil Gajjar | Powered by Streamlit & Machine Learning 💡")

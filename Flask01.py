import streamlit as st
import mlflow.pyfunc

# Load the model
model = mlflow.pyfunc.load_model("model")

st.title("Heart Disease Prediction")

# Input features
age = st.number_input("Age")
chol = st.number_input("Cholesterol")
trestbps = st.number_input("Resting Blood Pressure")
thalach = st.number_input("Maximum Heart Rate")
oldpeak = st.number_input("Oldpeak")

if st.button("Predict"):
    features = [[age, chol, trestbps, thalach, oldpeak]]
    prediction = model.predict(features)
    st.write("Prediction:", prediction[0])

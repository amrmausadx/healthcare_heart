import pandas as pd
import joblib  # Use joblib to load the pickle file
import streamlit as st

# Load the real and clean datasets
real_data = pd.read_csv('heart.csv')
clean_data = pd.read_csv('heart_cleaned.csv')

# Calculate min and max for each feature from real data
age_min, age_max = real_data['age'].min(), real_data['age'].max()
chol_min, chol_max = real_data['chol'].min(), real_data['chol'].max()
trestbps_min, trestbps_max = real_data['trestbps'].min(), real_data['trestbps'].max()
thalach_min, thalach_max = real_data['thalach'].min(), real_data['thalach'].max()
oldpeak_min, oldpeak_max = real_data['oldpeak'].min(), real_data['oldpeak'].max()

# Additional features that need to be included
sex_min, sex_max = real_data['sex'].min(), real_data['sex'].max() 
cp_min, cp_max = real_data['cp'].min(), real_data['cp'].max()
fbs_min, fbs_max = real_data['fbs'].min(), real_data['fbs'].max()
restecg_min, restecg_max = real_data['restecg'].min(), real_data['restecg'].max()
exang_min, exang_max = real_data['exang'].min(), real_data['exang'].max()
slope_min, slope_max = real_data['slope'].min(), real_data['slope'].max()
ca_min, ca_max = real_data['ca'].min(), real_data['ca'].max()
thal_min, thal_max = real_data['thal'].min(), real_data['thal'].max()

# Get min/max from the clean dataset
def normalize(value, real_min, real_max, clean_min, clean_max):
    return (value - real_min) / (real_max - real_min) * (clean_max - clean_min) + clean_min

# Load the model from the pkl file
model = joblib.load('model.pkl')  # Load the model here

st.title("Heart Disease Prediction")

# Input features with dynamic min and max values from real data
age = st.number_input(f"Age (Min: {age_min}, Max: {age_max})", min_value=int(age_min), max_value=int(age_max), value=int(age_min))
sex = st.number_input(f"Sex (0 = Female, 1 = Male)", min_value=int(sex_min), max_value=int(sex_max), value=int(sex_min))
cp = st.number_input(f"Chest Pain Type (Min: {cp_min}, Max: {cp_max})", min_value=int(cp_min), max_value=int(cp_max), value=int(cp_min))
trestbps = st.number_input(f"Resting Blood Pressure (Min: {trestbps_min}, Max: {trestbps_max})", min_value=int(trestbps_min), max_value=int(trestbps_max), value=int(trestbps_min))
chol = st.number_input(f"Cholesterol (Min: {chol_min}, Max: {chol_max})", min_value=int(chol_min), max_value=int(chol_max), value=int(chol_min))
fbs = st.number_input(f"Fasting Blood Sugar (0 = < 120 mg/dl, 1 = > 120 mg/dl)", min_value=int(fbs_min), max_value=int(fbs_max), value=int(fbs_min))
restecg = st.number_input(f"Resting ECG Results (Min: {restecg_min}, Max: {restecg_max})", min_value=int(restecg_min), max_value=int(restecg_max), value=int(restecg_min))
thalach = st.number_input(f"Maximum Heart Rate (Min: {thalach_min}, Max: {thalach_max})", min_value=int(thalach_min), max_value=int(thalach_max), value=int(thalach_min))
exang = st.number_input(f"Exercise Induced Angina (0 = No, 1 = Yes)", min_value=int(exang_min), max_value=int(exang_max), value=int(exang_min))
oldpeak = st.number_input(f"Oldpeak (Min: {oldpeak_min}, Max: {oldpeak_max})", min_value=float(oldpeak_min), max_value=float(oldpeak_max), value=float(oldpeak_min))
slope = st.number_input(f"Slope of the Peak Exercise ST Segment (Min: {slope_min}, Max: {slope_max})", min_value=int(slope_min), max_value=int(slope_max), value=int(slope_min))
ca = st.number_input(f"Number of Major Vessels Colored by Fluoroscopy (Min: {ca_min}, Max: {ca_max})", min_value=int(ca_min), max_value=int(ca_max), value=int(ca_min))
thal = st.number_input(f"Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)", min_value=int(thal_min), max_value=int(thal_max), value=1) #int(thal.min()))

if st.button("Predict"):
    try:
        # Normalize the input data based on the clean dataset's normalization
        age_normalized = normalize(age, age_min, age_max, clean_data['age'].min(), clean_data['age'].max())
        chol_normalized = normalize(chol, chol_min, chol_max, clean_data['chol'].min(), clean_data['chol'].max())
        trestbps_normalized = normalize(trestbps, trestbps_min, trestbps_max, clean_data['trestbps'].min(), clean_data['trestbps'].max())
        thalach_normalized = normalize(thalach, thalach_min, thalach_max, clean_data['thalach'].min(), clean_data['thalach'].max())
        oldpeak_normalized = normalize(oldpeak, oldpeak_min, oldpeak_max, clean_data['oldpeak'].min(), clean_data['oldpeak'].max())

        # Prepare the full feature set for prediction (normalized)
        features = pd.DataFrame({
            'age': [age_normalized],
            'sex': [float(sex)],
            'cp': [float(cp)],
            'trestbps': [trestbps_normalized],
            'chol': [chol_normalized],
            'fbs': [float(fbs)],
            'restecg': [float(restecg)],
            'thalach': [thalach_normalized],
            'exang': [float(exang)],
            'oldpeak': [oldpeak_normalized],
            'slope': [float(slope)],
            'ca': [float(ca)],
            'thal': [float(thal)]
        })

        # Make prediction
        prediction = model.predict(features)

        # Display the prediction result
        heart_disease_status = "Presence of heart disease" if prediction == 1 else "Absence of heart disease"
        st.write(f"Prediction result: {heart_disease_status}")

    except Exception as e:
        st.error(f"An error occurred: {e}")

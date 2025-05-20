import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('model.pkl')

st.title("Health Insurance Cost Predictor")

# Input fields
age = st.number_input("Age", 18, 100)
sex = st.selectbox("Sex", ['male', 'female'])
bmi = st.number_input("BMI", 10.0, 50.0)
children = st.number_input("Number of children", 0, 10)
smoker = st.selectbox("Smoker?", ['yes', 'no'])
region = st.selectbox("Region", ['northeast', 'northwest', 'southeast', 'southwest'])

# Preprocess inputs
sex_val = 0 if sex == 'female' else 1
smoker_val = 1 if smoker == 'yes' else 0
region_vals = {
    'northeast': [0, 0, 0],
    'northwest': [1, 0, 0],
    'southeast': [0, 1, 0],
    'southwest': [0, 0, 1],
}
region_data = region_vals[region]

# Prediction
if st.button("Predict"):
    input_data = np.array([[age, sex_val, bmi, children, smoker_val] + region_data])
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Insurance Cost: ${prediction:.2f}")

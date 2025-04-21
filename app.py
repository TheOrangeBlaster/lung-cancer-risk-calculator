# app.py
import streamlit as st
import joblib
import numpy as np

# Set this first — before any other Streamlit commands
st.set_page_config(page_title="Lung Cancer Prediction", layout="centered")

# Now the rest
st.write("✅ Streamlit app is running.")

# Load the model and label encoder
model, le_gender = joblib.load("cancer_model.pkl")

st.title("🩺 Lung Cancer Risk Predictor")
st.write("Fill in the details below to check your risk of developing lung cancer.")

# Input form
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 1, 100, 30)
smoking = st.selectbox("Do you smoke?", ["No", "Yes"])
yellow_fingers = st.selectbox("Yellow Fingers?", ["No", "Yes"])
anxiety = st.selectbox("Anxiety?", ["No", "Yes"])
peer_pressure = st.selectbox("Peer Pressure?", ["No", "Yes"])
chronic_disease = st.selectbox("Chronic Disease?", ["No", "Yes"])
fatigue = st.selectbox("Fatigue?", ["No", "Yes"])
allergy = st.selectbox("Allergy?", ["No", "Yes"])
wheezing = st.selectbox("Wheezing?", ["No", "Yes"])
alcohol = st.selectbox("Alcohol Consuming?", ["No", "Yes"])
coughing = st.selectbox("Coughing?", ["No", "Yes"])
short_breath = st.selectbox("Shortness of Breath?", ["No", "Yes"])
swallow_diff = st.selectbox("Swallowing Difficulty?", ["No", "Yes"])
chest_pain = st.selectbox("Chest Pain?", ["No", "Yes"])

# Encode gender
gender_encoded = le_gender.transform([gender[0]])[0]

# Prepare feature vector
features = np.array([[
    gender_encoded,
    age,
    smoking == "Yes",
    yellow_fingers == "Yes",
    anxiety == "Yes",
    peer_pressure == "Yes",
    chronic_disease == "Yes",
    fatigue == "Yes",
    allergy == "Yes",
    wheezing == "Yes",
    alcohol == "Yes",
    coughing == "Yes",
    short_breath == "Yes",
    swallow_diff == "Yes",
    chest_pain == "Yes"
]])

# Prediction logic
if st.button("Predict"):
    probability = model.predict_proba(features)[0][1]
    percentage = round(probability * 100, 2)

    st.write("### Prediction Result:")
    st.write(f"**Predicted Probability of Lung Cancer:** `{percentage}%`")
    
    if percentage >= 50:
        st.markdown("#### 🛑 High Risk of Lung Cancer")
    else:
        st.markdown("#### ✅ Low Risk of Lung Cancer")

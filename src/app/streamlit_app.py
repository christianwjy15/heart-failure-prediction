import streamlit as st
import pandas as pd
import requests


# Page Configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# App Title and Description
st.title("❤️ Heart Disease Prediction App")
st.markdown("""
This app uses a machine learning model to predict the likelihood of a patient having heart disease. 
Please enter the patient's information in the sidebar to get a prediction.

**Disclaimer:** This is for demonstration purposes only and should not replace expert medical consultation
""")

# Sidebar for User Inputs
st.sidebar.header("Patient Details")

def get_user_input():
    """Gets user input from the sidebar."""
    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=58, step=1)
    sex = st.sidebar.selectbox("Sex", ('M', 'F'))
    chest_pain_type = st.sidebar.selectbox("Chest Pain Type", ('ATA', 'NAP', 'ASY', 'TA'))
    resting_bp = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=136)
    cholesterol = st.sidebar.number_input("Serum Cholesterol (mg/dl)", min_value=50, max_value=600, value=150)
    fasting_bs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", (0, 1), format_func=lambda x: 'True' if x == 1 else 'False')
    resting_ecg = st.sidebar.selectbox("Resting ECG", ('Normal', 'ST', 'LVH'))
    max_hr = st.sidebar.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=152)
    exercise_angina = st.sidebar.selectbox("Exercise-Induced Angina", ('N', 'Y'))
    oldpeak = st.sidebar.slider("Oldpeak (ST depression)", min_value=0.0, max_value=7.0, value=0.0, step=0.1)
    st_slope = st.sidebar.selectbox("ST Slope", ('Up', 'Flat', 'Down'))

    # Create a dictionary from the inputs
    input_dict = {
        'Age': age,
        'Sex': sex,
        'ChestPainType': chest_pain_type,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'RestingECG': resting_ecg,
        'MaxHR': max_hr,
        'ExerciseAngina': exercise_angina,
        'Oldpeak': oldpeak,
        'ST_Slope': st_slope
    }
    return input_dict

# Get the user input
input_dict = get_user_input()

# Display User Input
input_df = pd.DataFrame([input_dict])
st.subheader("Patient's Input Data")
st.write(input_df)

# Prediction Logic
if st.sidebar.button("Predict"):


    response = requests.post("http://api:8000/predict", json=input_dict)

    # use this code if you don't want use Docker
    #response = requests.post("http://localhost:8000/predict", json=input_dict)

    try:
        prediction = response.json()["prediction"]
        probability = [response.json()["probability_normal"], response.json()["probability_heart_disease"]]
        
        if prediction is None:
            st.error("Prediction failed. Please check the logs. Have you trained the model and saved the artifacts?")
        else:
            st.subheader("Prediction Result")
            prob_heart_disease = probability[1]

            # Display with a metric card
            if prediction == 1:
                st.error(f"**Prediction: Heart Disease**", icon="💔")
            else:
                st.success(f"**Prediction: Normal**", icon="💚")

            st.metric(
                label="Probability of Heart Disease",
                value=f"{prob_heart_disease:.2%}",
            )

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

else:
    st.info("Click the 'Predict' button in the sidebar to see the result.")
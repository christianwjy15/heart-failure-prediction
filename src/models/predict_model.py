import pandas as pd
import numpy as np
import joblib
import yaml
import os

def load_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)

def predict_new_data(model, input_data: pd.DataFrame):
    # Create a copy to avoid modifying the original DataFrame
    data = input_data.copy()
    
    # Preprocessing Steps
    # Features Encoding
    data['Sex'] = data['Sex'].map({'M': 1, 'F': 0})
    data['ExerciseAngina'] = data['ExerciseAngina'].map({'Y': 1, 'N': 0})
    data['ST_Slope'] = data['ST_Slope'].map({'Down': 0, 'Flat': 1, 'Up': 2})
    data['ChestPainType'] = data['ChestPainType'].map({'ASY': 0, 'NAP': 1, 'ATA': 2, 'TA': 3})
    data['RestingECG'] = data['RestingECG'].map({'Normal': 0, 'ST': 1, 'LVH': 2})

    # Make Prediction
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0]

    return prediction, probability

if __name__ == "__main__":
    # Load the configuration
    config = load_config()

    # Create Sample New Data for Prediction
    new_patient_data = {
        'Age': 58,
        'Sex': 'M',
        'ChestPainType': 'ATA',
        'RestingBP': 136,
        'Cholesterol': 319, 
        'FastingBS': 0,
        'RestingECG': 'ST',
        'MaxHR': 152,
        'ExerciseAngina': 'N',
        'Oldpeak': 0.0,
        'ST_Slope': 'Up'
    }

    # Convert the dictionary to a DataFrame
    input_df = pd.DataFrame([new_patient_data])

    # load model
    loaded_model = joblib.load(config['model']['output_path'])
    
    # Get the prediction
    prediction, probability = predict_new_data(loaded_model, input_df)

    # --- Display the Result ---
    if prediction is not None:
        print("\n--- Prediction Result ---")
        #print(f"Input Data:\n{input_df.to_string(index=False)}")
        #print("-" * 25)
        print(f"Predicted Class: {prediction} ({'Heart Disease' if prediction == 1 else 'Normal'})")
        print(f"Prediction Probability (0: Normal, 1: Heart Disease): {probability}")
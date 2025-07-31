from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import yaml
import joblib
from src.models.predict_model import predict_new_data


app = FastAPI(
    title="Heart Disease Prediction API",
)

# Define the Input Data Model using Pydantic
class HeartDiseaseInput(BaseModel):
    Age: int 
    Sex: str 
    ChestPainType: str 
    RestingBP: int 
    Cholesterol: int 
    FastingBS: int 
    RestingECG: str 
    MaxHR: int 
    ExerciseAngina: str 
    Oldpeak: float 
    ST_Slope: str 

    class Config:
        schema_extra = {
            "example": {
                "Age": 58,
                "Sex": "M",
                "ChestPainType": "ATA",
                "RestingBP": 136,
                "Cholesterol": 319,
                "FastingBS": 0,
                "RestingECG": "ST",
                "MaxHR": 152,
                "ExerciseAngina": "N",
                "Oldpeak": 0.0,
                "ST_Slope": "Up"
            }
        }

# Load model
config = yaml.safe_load(open("config.yaml"))
model = joblib.load(config['model']['output_path'])


# Define API Endpoints
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Heart Disease Prediction API is running!"}

@app.post("/predict")
def predict(patient_data: HeartDiseaseInput):
    # Convert the Pydantic model to a pandas DataFrame
    input_df = pd.DataFrame([patient_data.model_dump()])
    
    try:
        prediction, probability = predict_new_data(model, input_df)

        if prediction is None:
             raise HTTPException(status_code=500, detail="Model prediction failed. Check server logs for details.")

        return {
            "prediction": int(prediction),
            "prediction_label": "Heart Disease" if prediction == 1 else "Normal",
            "probability_normal": float(probability[0]),
            "probability_heart_disease": float(probability[1])
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")

# uvicorn src.app.api:app --reload     
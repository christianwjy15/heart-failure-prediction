# End-to-End Heart Disease Prediction ❤️
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 310](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

This project is an end-to-end machine learning application to predict the likelihood of heart disease based on patient data. It use MLOps workflow, from data preprocessing and model training to deployment as a containerized application.

---

## Features

- **Model Training & Tuning**: Trains multiple models (Random Forest, XGBoost, LightGBM, CatBoost) and uses Optuna for hyperparameter optimization.
- **Experiment Tracking**: Integrates with MLflow to log experiments, parameters, and metrics.
- **Containerization**: Fully containerized using Docker and orchestrated with Docker Compose for easy setup and deployment.
- **Interactive UI**: A user-friendly dashboard to input patient data and receive real-time predictions.

* **Data Preprocessing:** Handles missing values and categorizes features.
* **Model Training:** Trains a [e.g., Random Forest Regressor] on the processed data.
* **Model Evaluation:** Reports key metrics like RMSE and MAE.
* **Interactive Demo (Optional):** A simple web application built with [e.g., Streamlit] for predictions.


---
## Project Structure

Briefly explain the directory structure of your repository. This helps users quickly find relevant files. For example:
---

## Getting Started
### Prerequisites
### Installation
### Usage

---
## Data
Describe the dataset(s) used in your project.

Source: Where did the data come from? (e.g., Kaggle, UCI, internal database)

Description: What does the data represent? What are the key features?

Preprocessing: Briefly mention any significant preprocessing steps applied.

Access: If the data is not included in the repo (due to size or privacy), explain how users can obtain it.

---
## Model
Provide details about the machine learning model(s) used.

Algorithm: What algorithm did you use (e.g., Logistic Regression, Random Forest, TensorFlow, PyTorch)?

Architecture: If it's a deep learning model, describe its architecture.

Training: Briefly explain the training process (e.g., epochs, batch size, optimizer).

Evaluation: What metrics did you use to evaluate the model's performance? Show key results if possible.

---
## Results
Summarize the performance of your model. This is where you can showcase graphs, tables, or key metrics.

Accuracy, Precision, Recall, F1-score, RMSE, MAE, etc.

Confusion matrices, ROC curves (if applicable, you can link to an image).

Insights gained from the model.

---
## Deployment (If Applicable)
If you've deployed your model, explain where it's deployed and how to access it.

Platform used (e.g., Heroku, AWS Sagemaker, Google Cloud AI Platform, Streamlit Cloud)

Link to the live application.

Instructions for deploying it yourself.

---

## Tech Stack
- **Modeling & Data Science**: Python, Pandas, Scikit-learn, XGBoost, LightGBM, CatBoost
- **MLOps & Tools**: MLflow, Optuna, Docker, Docker Compose
- **API & UI**: FastAPI, Uvicorn, Streamlit, Requests


python -m src.data.data_preprocessing
python -m src.models.train_model
python -m src.models.evaluate_model
mlflow ui

docker-compose up --build
buka localhost

atau

uvicorn src.app.api:app --reload

ubah code streamlit
streamlit run src/app/streamlit_app.py
buka localhost

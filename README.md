# End-to-End Heart Disease Prediction ❤️

This project is an end-to-end machine learning application to predict the likelihood of heart disease based on patient data. It use MLOps workflow, from data preprocessing and model training to deployment as a containerized application.

---

## Features

- **Model Training & Tuning**: Trains multiple models (Random Forest, XGBoost, LightGBM, CatBoost) and uses Optuna for hyperparameter optimization.
- **Experiment Tracking**: Integrates with MLflow to log experiments, parameters, and metrics.
- **Containerization**: Fully containerized using Docker and orchestrated with Docker Compose for easy setup and deployment.
- **Interactive UI**: A user-friendly dashboard to input patient data and receive real-time predictions.

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

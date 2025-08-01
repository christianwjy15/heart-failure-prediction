# End-to-End Heart Disease Prediction ❤️
[![Python 310](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

This project is an end-to-end machine learning application to predict the likelihood of heart disease based on patient data. It use MLOps workflow, from data preprocessing and model training to deployment as a containerized application.

---

## Features

- **Model Training & Tuning**: Trains multiple models (Random Forest, XGBoost, LightGBM, CatBoost) and uses Optuna for hyperparameter optimization.
- **Experiment Tracking**: Integrates with MLflow to log experiments, parameters, and metrics.
- **Decoupled Architecture**: Deployed as two distinct services:
  -  A **FastAPI** backend that serves the ML model via a REST API.
  -  A **Streamlit** frontend that provides an interactive user interface for making predictions.
- **Containerization**: Fully containerized using Docker and orchestrated with Docker Compose for easy setup and deployment.
- **Interactive UI**: A user-friendly dashboard to input patient data and receive real-time predictions.
  
---

## Tech Stack
- **Modeling & Data Science**: Python, Pandas, Scikit-learn, XGBoost, LightGBM, CatBoost
- **MLOps & Tools**: MLflow, Optuna, Docker, Docker Compose
- **API & UI**: FastAPI, Uvicorn, Streamlit, Requests

---

## Project Structure
``` bash
.
├── .dockerignore                  # Specifies files and directories to ignore when building Docker images.
├── .gitignore                     # Defines untracked files and directories to be ignored by Git.
├── config.yaml                    # Configuration file for various project settings.
├── docker-compose.yml             # Defines and runs multi-container Docker applications (FastAPI and Streamlit).
├── Dockerfile.fastapi             # Dockerfile for building the FastAPI application image.
├── Dockerfile.streamlit           # Dockerfile for building the Streamlit application image.
├── README.md                      # File for providing an overview of the project.
├── requirements.txt               # Primary Python dependencies for the entire project.
├── requirements_fastapi.txt       # Specific Python dependencies for the FastAPI service.
├── requirements_streamlit.txt     # Specific Python dependencies for the Streamlit application.
│
├── data/                          # Stores all project data.
│   ├── processed/                 # Contains cleaned and preprocessed data ready for model training/evaluation.
│   │   ├── test.csv               # Processed test dataset.
│   │   └── train.csv              # Processed training dataset.
│   └── raw/                       # Holds original, unprocessed datasets.
│       └── heart.csv              # Raw heart disease dataset.
│
├── models/                        # Contains trained machine learning models and related artifacts.
│   ├── best_model_params.json     # Stores parameters of the best performing model.
│   └── final_model.joblib         # The final trained machine learning model.
│
├── notebooks/                     # Jupyter notebooks for exploratory data analysis (EDA) and experimentation.
│   ├── data_preprocessing.ipynb   # Notebook for initial data cleaning and transformation.
│   ├── eda.ipynb                  # Notebook for exploratory data analysis and visualization.
│   └── modeling.ipynb             # Notebook for model training, evaluation, prediction and hyperparameter tuning experiments.
│
└── src/                           # Source code directory, housing all core application logic.
    ├── __init__.py                # Marks 'src' as a Python package.
    │
    ├── app/                       # Contains the main application services.
    │   ├── api.py                 # Defines the FastAPI application endpoints.
    │   └── streamlit_app.py       # The Streamlit web application interface.
    │
    ├── data/                      # Modules for data handling and processing.
    │   ├── data_loader.py         # Script to load raw data from various sources.
    │   ├── data_preprocessing.py  # Script for data cleaning, transformation, and feature engineering.
    │   └── __init__.py            # Marks 'src/data' as a Python package.
    │
    └── models/                    # Modules for model development and management.
        ├── evaluate_model.py      # Script to evaluate trained models.
        ├── predict_model.py       # Script for making predictions using a trained model.
        ├── train_model.py         # Script to train and save machine learning models.
        └── __init__.py            # Marks 'src/models' as a Python package.
```

---

## Getting Started
### Prerequisites
To run this project, ensure you have the following tools installed:

- **Python 3.10+**
- **Git**
- **Docker**
- **Docker Compose**

> **Note:** If you prefer not to use Docker, you can still run the application locally with minor modifications. Detailed instructions are provided in the **Usage** section.

---

### Installation
#### Clone the Repository
```bash
git clone https://github.com/christianwjy15/heart-failure-prediction.git
cd heart-failure-prediction
```

#### Create and Activate a Virtual Environment
``` bash
python -m venv .venv
source .venv/bin/activate  # On Windows: `.venv\Scripts\activate`
```

#### Install Dependencies
``` bash
pip install -r requirements.txt
```

---

### Usage
#### 1. Preprocess Data, Train, and Evaluate Model
Run the following commands to preprocess the data, train the model, and evaluate its performance:

``` bash
python -m src.data.data_preprocessing
python -m src.models.train_model
python -m src.models.evaluate_model

```
#### 2. Launch MLflow UI (Optional for Experiment Tracking)
```bash
mlflow ui
```
Open http://localhost:5000 in your browser to explore experiment runs, parameters, metrics, and artifacts.

#### 3. Run the API and Streamlit App
##### *If Using Docker (Recommended)*
``` bash
docker-compose up --build
```
This builds and starts both the FastAPI and Streamlit services inside containers.


##### *If Not Using Docker*
If you prefer to run the app without Docker, you’ll need to modify the API endpoint in the Streamlit app. Edit the file: ```src/app/streamlit_app.py``` 

Change this line:
``` bash
response = requests.post("http://api:8000/predict", json=input_dict)
```

To:
``` bash
response = requests.post("http://localhost:8000/predict", json=input_dict)
```

Then, run the services manually:

Start the FastAPI backend:
``` bash
uvicorn src.app.api:app --reload
```

Start the Streamlit frontend:
``` bash
streamlit run src/app/streamlit_app.py
```

#### 4. Access the Applications
Once the services are running, open your browser to:

- **Streamlit Dashboard**: http://localhost:8501

- **FastAPI Docs (Swagger UI)**: http://localhost:8000/docs


---

## Data

This project uses the [Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) dataset from Kaggle. The dataset contains **918 patient records** and **11 features**, which are used to predict the likelihood of heart disease.

### Features

- **Age**: Age of the patient (in years)  
- **Sex**: Biological sex (`M`: Male, `F`: Female)  
- **ChestPainType**: Type of chest pain  
  - `TA`: Typical Angina  
  - `ATA`: Atypical Angina  
  - `NAP`: Non-Anginal Pain  
  - `ASY`: Asymptomatic  
- **RestingBP**: Resting blood pressure (mm Hg)  
- **Cholesterol**: Serum cholesterol (mg/dl)  
- **FastingBS**: Fasting blood sugar (`1` if >120 mg/dl, else `0`)  
- **RestingECG**: Resting electrocardiogram results  
  - `Normal`: Normal  
  - `ST`: ST-T wave abnormality  
  - `LVH`: Left ventricular hypertrophy  
- **MaxHR**: Maximum heart rate achieved (60–202)  
- **ExerciseAngina**: Exercise-induced angina (`Y`: Yes, `N`: No)  
- **Oldpeak**: ST depression induced by exercise  
- **ST_Slope**: Slope of the peak exercise ST segment (`Up`, `Flat`, `Down`)  

### Target

- **HeartDisease**: Output class (`1`: Heart Disease, `0`: Normal)

### Preprocessing

The dataset is relatively clean (no missing or duplicate values). The following preprocessing steps were applied:

- Removed or corrected invalid data points (e.g., impossible values)
- Encoded categorical features using suitable techniques
- Split the dataset into **train and test sets** using **stratified sampling** (80% train / 20% test)

> Stratification ensures that the distribution of the target variable (`HeartDisease`) is preserved in both the training and testing datasets.


---
## Model

For model training, four different machine learning algorithms were evaluated:

- **Random Forest**
- **XGBoost**
- **LightGBM**
- **CatBoost**

### Evaluation Strategy

- **Cross-Validation**: Stratified K-Fold Cross Validation (`k=5`) was used to ensure balanced distribution of the target class across folds.
- **Hyperparameter Tuning**: Optuna was used for automated hyperparameter optimization, with:
  - `n_trials = 50`
  - Objective metric: **AUC (Area Under the ROC Curve)**

### Best Performing Model

After tuning and evaluation, **CatBoost** achieved the highest AUC score and was selected as the final model for deployment.

> AUC was chosen as the optimization metric due to its effectiveness in measuring the model’s ability to distinguish between the positive and negative classes.


---

## Results

After identifying **CatBoost** as the best model through cross-validation and hyperparameter tuning, it was retrained on the **full training set** using the best-found parameters. The model was then evaluated on the test set.

### Test Set Performance

- **AUC**: `0.93` — Excellent ability to distinguish between heart disease and normal cases.
- **Recall**: `0.90` — The model correctly identified 90% of actual heart disease cases.
- **Precision**: `0.91` — When the model predicts heart disease, it is correct 91% of the time.
- **F1-Score**: `0.91` — A balanced metric indicating strong performance across both recall and precision.


### Confusion Matrix
<img width="480" height="360" alt="image" src="https://github.com/user-attachments/assets/6f455997-8a60-4e3c-9ebc-73c6938449e9" />

This means:
- **73** patients without heart disease were correctly classified
- **93** patients with heart disease were correctly classified  
- **9** healthy patients were incorrectly predicted to have heart disease (False Positives)  
- **9** patients with heart disease were missed by the model (False Negatives)



### Insights

- The **high AUC (0.93)** confirms that the model has strong discriminative power.
- **High recall (0.90)** ensures that most heart disease cases are detected — crucial for real-world medical applications.
- **Balanced precision and recall** indicate a low number of false alarms and misses.
- The **confusion matrix shows low misclassification**, making the model dependable in distinguishing between positive and negative cases.

> In health-related models, **recall is critical** to avoid missing true cases of heart disease. This model demonstrates a strong ability to achieve that.











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
##### If Using Docker (Recommended)
``` bash
docker-compose up --build
```
This builds and starts both the FastAPI and Streamlit services inside containers.

#### If Not Using Docker
If you prefer to run the app without Docker, you’ll need to modify the API endpoint in the Streamlit app.
Edit the file: ```src/app/streamlit_app.py``` 
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

Streamlit Dashboard: http://localhost:8501

FastAPI Docs (Swagger UI): http://localhost:8000/docs


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





import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.datasets import make_classification
import yaml
import joblib
import json
import mlflow
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from src.data.data_loader import load_preprocessed_data

def load_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)
    

# Objective Function for Optuna & MLflow
def objective(trial, model_name):
    """Objective function to tune hyperparameters and log results to MLflow."""
    # Start an MLflow run for each trial
    with mlflow.start_run(nested=True):
        mlflow.set_tag("model_name", model_name)
        
        # Define hyperparameter search space
        if model_name == 'RandomForest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'random_state': 42,
            }
            model = RandomForestClassifier(**params)

        elif model_name == 'XGBoost':
            params = {
                'objective': 'binary:logistic', 'eval_metric': 'auc',
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': 42,
            }
            model = XGBClassifier(**params)

        elif model_name == 'LightGBM':
            params = {
                'objective': 'binary', 'metric': 'auc', 'verbosity': -1,
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'random_state': 42,
            }
            model = LGBMClassifier(**params)

        elif model_name == 'CatBoost':
            params = {
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'depth': trial.suggest_int('depth', 3, 10),
                'verbose': 0, 'random_seed': 42,
            }
            model = CatBoostClassifier(**params)
        
        # Log parameters to MLflow
        mlflow.log_params(params)
        
        # Evaluate using cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        score = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
        
        # Log the score to MLflow
        mlflow.log_metric("mean_cv_auc", score)
        
    return score


def model_training(n_trials=50):
    # Run Optimization and Store Results
    models_to_tune = ['RandomForest', 'XGBoost', 'LightGBM', 'CatBoost']
    best_models = {}

    mlflow.set_experiment("Heart Diseases Classification")

    for model_name in models_to_tune:
        with mlflow.start_run(run_name=f"Tuning_{model_name}", description=f"Optimizing {model_name} with Optuna"):
            print(f"Tuning {model_name}...") 
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: objective(trial, model_name), n_trials=n_trials)
            
            best_models[model_name] = {
                'best_score': study.best_value,
                'best_params': study.best_params,
                #'study': study
            }
            mlflow.log_metric("best_cv_auc", study.best_value)
            mlflow.log_params(study.best_params)
            print(f"Best CV AUC for {model_name}: {study.best_value:.4f}")
            print("-" * 30)
    
    # save best model parameters in json format
    file_path = "models/best_model_params.json"
    with open(file_path, 'w') as json_file:
        json.dump(best_models, json_file, indent=4)
        

def save_best_model(model_params, path="models/final_model.joblib"):
    best_overall_model_name = max(model_params, key=lambda k: model_params[k]['best_score'])
    best_params = model_params[best_overall_model_name]['best_params']
    print(f"\nBest overall model is: {best_overall_model_name}")

    if best_overall_model_name == 'RandomForest':
        final_model = RandomForestClassifier(**best_params, random_state=42)
    elif best_overall_model_name == 'XGBoost':
        final_model = XGBClassifier(**best_params, use_label_encoder=False, random_state=42)
    elif best_overall_model_name == 'LightGBM':
        final_model = LGBMClassifier(**best_params, random_state=42)
    else: # CatBoost
        final_model = CatBoostClassifier(**best_params, verbose=0, random_seed=42)
    
    final_model.fit(X_train, y_train)
    joblib.dump(final_model, path)


if __name__ == "__main__":
    X_train, y_train, _, _ = load_preprocessed_data()
    model_training()
    with open("models/best_model_params.json", 'r') as json_file:
        model_params = json.load(json_file)
    save_best_model(model_params)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import joblib
import json
import mlflow
from src.data.data_loader import load_preprocessed_data


def load_model(model_path="models/final_model.joblib"):
    return joblib.load(model_path)

def evaluate_model(model_params, final_model):
    best_overall_model_name = max(model_params, key=lambda k: model_params[k]['best_score'])
    best_params = model_params[best_overall_model_name]['best_params']

    mlflow.set_experiment("Heart Diseases Classification")
    # Log the final, best model and its test performance in a new run
    with mlflow.start_run(run_name="Final_Best_Model"):
        print("\nEvaluating Final Model")

        y_pred = final_model.predict(X_test)
        y_pred_proba = final_model.predict_proba(X_test)[:, 1]

        # Calculate final metrics
        test_auc = roc_auc_score(y_test, y_pred_proba)
        test_precision = precision_score(y_test, y_pred)
        test_recall = recall_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred)
        
        # Log parameters, metrics, and tags to MLflow
        mlflow.set_tag("best_model", best_overall_model_name)
        mlflow.log_params(best_params)
        mlflow.log_metrics({
            "test_auc": test_auc,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1_score": test_f1
        })

        # Log the confusion matrix as an image artifact
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        mlflow.log_figure(fig, "confusion_matrix.png")
        
        # Log the model artifact
        mlflow.sklearn.log_model(final_model, name=best_overall_model_name)
        
        print(f"Final Model: {best_overall_model_name}")
        print(f"Test AUC: {test_auc:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test F1-Score: {test_f1:.4f}")
        print("\nFinal model, metrics, and artifacts logged to MLflow.")

if __name__ == "__main__":
    _, _, X_test, y_test = load_preprocessed_data()
    with open("models/best_model_params.json", 'r') as json_file:
        model_params = json.load(json_file)
    
    model = load_model()
    evaluate_model(model_params, model)
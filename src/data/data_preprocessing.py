import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import yaml
import os
from src.data.data_loader import load_data

def load_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)

def preprocess_data(config):
    df = load_data(config['data']['raw_path'])

    # Handle Missing Value
    df.dropna(inplace=True)

    # Handle Duplicate Data
    df.drop_duplicates(inplace=True)

    # Handle Invalid Data
    imputer = SimpleImputer(strategy='median')
    df['Cholesterol'] = df['Cholesterol'].replace(0, np.nan)
    df[['Cholesterol']] = imputer.fit_transform(df[['Cholesterol']])

    # Features Encoding
    df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})
    df['ExerciseAngina'] = df['ExerciseAngina'].map({'Y': 1, 'N': 0})
    df['ST_Slope'] = df['ST_Slope'].map({'Down': 0, 'Flat': 1, 'Up': 2})
    df['ChestPainType'] = df['ChestPainType'].map({'ASY': 0, 'NAP': 1, 'ATA': 2, 'TA':3})
    df['RestingECG'] = df['RestingECG'].map({'Normal': 0, 'ST': 1, 'LVH': 2})

    # split data
    X = df.drop(config['data']['target'], axis=1)
    y = df[config['data']['target']]  

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['data']['test_size'], stratify=y, random_state=config['data']['random_state']
    ) 

    # Scale Data
    # I don't scale the data because I use tree-based model

    # Save Splitted Data
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # Save to CSV
    train_df.to_csv(os.path.join(config["data"]["processed_dir"], "train.csv"), index=False)
    test_df.to_csv(os.path.join(config["data"]["processed_dir"], "test.csv"), index=False)

if __name__ == "__main__":
    config = load_config()
    preprocess_data(config)

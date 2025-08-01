import pandas as pd
import yaml

def load_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)

def load_data(data_path):
    return pd.read_csv(data_path)
    
def load_preprocessed_data(train_path="data/processed/train.csv", test_path="data/processed/test.csv"):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    X_train = df_train.drop('HeartDisease', axis=1)
    y_train = df_train['HeartDisease']

    X_test = df_test.drop('HeartDisease', axis=1)
    y_test = df_test['HeartDisease']

    return X_train, y_train, X_test, y_test
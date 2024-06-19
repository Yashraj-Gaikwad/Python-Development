# scripts/preprocess.py

import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    # Example preprocessing: dropping NA values and encoding categorical variables
    df = df.dropna()
    df = pd.get_dummies(df, drop_first=True)
    return df

def split_data(df, target_column):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = load_data('data/raw/customer_data.csv')
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df, 'churn')
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)



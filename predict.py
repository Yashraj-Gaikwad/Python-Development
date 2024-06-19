# scripts/predict.py

import joblib
import pandas as pd

def load_model(filepath):
    return joblib.load(filepath)

def predict(model, X):
    return model.predict(X)

if __name__ == "__main__":
    model = load_model('models/customer_churn_model.pkl')
    X_test = pd.read_csv('data/processed/X_test.csv')
    predictions = predict(model, X_test)
    print(predictions)



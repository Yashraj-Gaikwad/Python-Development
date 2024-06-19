# scripts/train.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def load_data():
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    
    lr_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    
    lr_accuracy = evaluate_model(lr_model, X_test, y_test)
    rf_accuracy = evaluate_model(rf_model, X_test, y_test)
    
    print(f"Logistic Regression Accuracy: {lr_accuracy}")
    print(f"Random Forest Accuracy: {rf_accuracy}")
    
    # Save the best model
    best_model = lr_model if lr_accuracy > rf_accuracy else rf_model
    joblib.dump(best_model, 'models/customer_churn_model.pkl')



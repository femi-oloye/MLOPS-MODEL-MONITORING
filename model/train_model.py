import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# Load Dataset
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
    
    # Feature Engineering
    df = df[["Pclass", "Sex", "Age", "Fare", "Survived"]].dropna()
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})  # Convert categorical to numeric
    
    X = df[["Pclass", "Sex", "Age", "Fare"]]
    y = df["Survived"]
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Model
def train_and_log_model():
    X_train, X_test, y_train, y_test = load_data()
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    # Calculate Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Log to MLflow
    mlflow.set_tracking_uri("http://localhost:6000")  # Ensure MLflow is running
    mlflow.set_experiment("Titanic-Survival-Prediction")

    with mlflow.start_run():
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Fix: Provide input example
        mlflow.sklearn.log_model(model, "model", input_example=X_train.iloc[:1])
    
    print(f"Model trained and logged successfully! Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    train_and_log_model()

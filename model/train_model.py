import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import os
import subprocess
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# ✅ Set MLflow Tracking URI
MLFLOW_TRACKING_URI = "http://54.161.192.202:6001"  # Ensure MLflow is running
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Titanic-Survival-Prediction")

# ✅ Load Dataset
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
    
    # Feature Engineering
    df = df[["Pclass", "Sex", "Age", "Fare", "Survived"]].dropna()
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})  # Convert categorical to numeric
    
    X = df[["Pclass", "Sex", "Age", "Fare"]]
    y = df["Survived"]
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ✅ Train Model
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
    
    with mlflow.start_run() as run:
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # ✅ Log Model
        model_info = mlflow.sklearn.log_model(model, "model", input_example=X_train.iloc[:1])
        model_uri = model_info.model_uri  # ✅ Extract Model URI
        
        # ✅ Register the model
        mlflow.register_model(model_uri, "Titanic-Survival-Prediction")
    
    print(f"✅ Model trained & logged successfully! Accuracy: {accuracy:.2f}")
    
    # ✅ Perform Model Drift Monitoring
    check_model_drift(X_train)

# ✅ Model Drift Detection
def check_model_drift(reference_data):
    # ✅ Simulate new incoming data
    new_data = reference_data.sample(frac=0.1, random_state=42).reset_index(drop=True)
    
    # ✅ Generate Data Drift Report
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference_data, current_data=new_data)
    
    # ✅ Extract Drift Summary
    drift_data = drift_report.as_dict()
    
    # ✅ Extract dataset drift metric
    try:
        drift_detected = drift_data["metrics"][0]["result"]["dataset_drift"]
        drift_percentage = drift_data["metrics"][0]["result"]["share_drifted_features"] * 100
    except KeyError:
        drift_detected = False
        drift_percentage = 0
    
    print(f"📊 Drift Detected: {drift_detected}, Drift Percentage: {drift_percentage:.2f}%")
    
    # ✅ Save Drift Report
    os.makedirs("monitoring", exist_ok=True)
    report_path = "monitoring/drift_report.html"
    drift_json_path = "monitoring/model_drift.json"
    drift_report.save_html(report_path)
    
    # ✅ Save drift data as JSON
    with open(drift_json_path, "w") as f:
        json.dump(drift_data, f)
    
    # ✅ Log Drift Report to MLflow
    with mlflow.start_run():
        mlflow.log_artifact(report_path, artifact_path="drift_reports")
        mlflow.log_artifact(drift_json_path, artifact_path="drift_reports")
        mlflow.log_metric("data_drift_detected", int(drift_detected))
        mlflow.log_metric("data_drift_percentage", drift_percentage)

    # 🚀 **Trigger Model Retraining if Drift > 30%**
    DRIFT_THRESHOLD = 30
    if drift_percentage > DRIFT_THRESHOLD:
        print(f"⚠️ WARNING: Data drift {drift_percentage:.2f}% exceeds threshold ({DRIFT_THRESHOLD}%)! Retraining model...")
        subprocess.run(["python", "train_model.py"], check=True)
    else:
        print(f"✅ No significant drift detected. Current drift: {drift_percentage:.2f}%")

if __name__ == "__main__":
    train_and_log_model()

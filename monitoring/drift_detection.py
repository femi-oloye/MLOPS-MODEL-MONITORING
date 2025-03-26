import os
import pandas as pd
import numpy as np
import subprocess
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import mlflow

# MLflow Tracking
mlflow.set_tracking_uri("http://localhost:6001")

# Load reference data (used during model training)
reference_data = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
reference_data = reference_data[["Pclass", "Sex", "Age", "Fare"]].dropna()
reference_data["Sex"] = reference_data["Sex"].map({"male": 0, "female": 1})

# Simulate new incoming data (random sampling)
new_data = reference_data.sample(frac=0.1).reset_index(drop=True)

# Generate Data Drift Report
drift_report = Report(metrics=[DataDriftPreset()])
drift_report.run(reference_data=reference_data, current_data=new_data)

# Extract Drift Summary
drift_data = drift_report.as_dict()
drift_detected = drift_data["metrics"][0]["result"]["dataset_drift"]

# Save Report to File
report_path = "monitoring/drift_report.html"
drift_report.save_html(report_path)

# ✅ Verify if file exists before logging to MLflow
if os.path.exists(report_path):
    print(f"✅ {report_path} exists! Proceeding to log it in MLflow.")
else:
    print(f"❌ ERROR: {report_path} does NOT exist! Check Evidently save function.")
    exit(1)  # Stop execution if the report is missing

# Start MLflow Run and Log Drift Report
with mlflow.start_run() as run:
    print(f"Logging artifact to MLflow run: {run.info.run_id}")
    mlflow.log_artifact(report_path, artifact_path="drift_reports")

print(f"Drift detected: {drift_detected}")

# 🚀 Trigger Model Retraining if Drift is Detected
if drift_detected:
    print("🔄 Data drift detected! Triggering model retraining...")
    subprocess.run(["python", "model/train_model.py"], check=True)
else:
    print("✅ No significant data drift detected.")

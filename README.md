# 🚀 MLOps Model Monitoring & Drift Detection

📌 Author: oluwafemi oloye
📌 GitHub Repository: https://github.com/femi-oloye/MLOPS-MODEL-MONITORING.git

📊 Project Overview

This project implements an end-to-end MLOps pipeline for model monitoring and drift detection using MLflow, Evidently AI, and Airflow.

🔹 Why? In production, ML models degrade due to data drift, impacting accuracy. This system monitors drift and triggers automatic model retraining when needed.

🔹 Key Outcomes:
✅ Automated model monitoring with Evidently AI & MLflow
✅ Airflow DAGs automate drift detection & model retraining
✅ MLflow tracks model versions & performance metrics
✅ FastAPI serves ML models via an API
✅ Dockerized setup for easy deployment

## 🛠 Tech Stack

✅ MLOps & Automation: Apache Airflow, MLflow, CI/CD
✅ Model Monitoring: Evidently AI (Drift Detection)
✅ Cloud & Infrastructure: Docker, AWS (optional)
✅ Machine Learning: Scikit-Learn, RandomForest
✅ API Deployment: FastAPI, Uvicorn

## 📌 Features

🔹 📡 Real-time Model Monitoring – Detects drift using Evidently AI
🔹 🔄 Automatic Model Retraining – Airflow triggers new training when drift is detected
🔹 📈 ML Model Tracking – MLflow logs model versions & performance
🔹 ⚙️ Automated Workflow – Airflow schedules drift detection daily
🔹 🌐 Model Deployment – FastAPI serves predictions via an API

## 📸 Screenshots

💻 Setup & Installation
1️⃣ Clone the Repository

```
git clone https://github.com/your-username/mlops-model-monitoring.git
cd mlops-model-monitoring
```
2️⃣ Install Dependencies
```
pip install -r requirements.txt
```
3️⃣ Start MLflow Server
```
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts --host 0.0.0.0 --port 5000
```
🔗 Access MLflow UI at: http://localhost:5000
4️⃣ Start FastAPI for Model Inference
```
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
🔗 API available at: http://localhost:8000/docs
5️⃣ Run Drift Detection Manually
```
python monitoring/drift_detection.py
```
🔗 Check drift_report.html for drift analysis
6️⃣ Start Airflow for Automated Monitoring
```
airflow scheduler & airflow webserver -p 8089
```
🔗 Access Airflow UI at: http://localhost:8080
✅ Enable & Trigger the DAG → drift_monitoring_and_retraining
📌 How It Works

1️⃣ Model Training & Logging

    Trains a RandomForestClassifier on the Titanic dataset

    Logs model & metrics in MLflow

2️⃣ Drift Detection with Evidently AI

    Compares new data vs. training data

    Generates drift report & logs it in MLflow

3️⃣ Airflow Automates Everything

    Runs daily drift detection

    If drift is detected → triggers model retraining

4️⃣ Model Deployment & API

    FastAPI serves model predictions

    MLflow tracks all model versions

🚀 Future Improvements

✅ Deploy full pipeline on AWS (S3, Lambda, EC2, RDS)
✅ Add alerting system (Slack, Email) for drift detection
✅ Integrate real-time streaming data sources
🤝 Contributing

🔹 Pull requests are welcome! Feel free to submit feature requests or report issues in the GitHub Issues section.
📞 Contact & Support

📩 Email: oluwafemi.ezra@gmail.com
🔗 LinkedIn: www.linkedin.com/in/oluwafemi-oloye-a3b772353
📜 License

This project is open-source and available under the MIT License.
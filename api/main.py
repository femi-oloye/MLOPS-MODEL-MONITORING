from fastapi import FastAPI
import mlflow
import pandas as pd
import joblib

app = FastAPI()

# MLflow Tracking URI
mlflow.set_tracking_uri("http://localhost:6001")

# Load the latest trained model from MLflow
model_name = "Titanic-Survival-Prediction"
model_version = "latest"

logged_model = f"models:/{model_name}/{model_version}"
model = mlflow.sklearn.load_model(logged_model)

@app.get("/")
def read_root():
    return {"message": "ML Model API is running!"}

@app.post("/predict/")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {"prediction": int(prediction[0])}

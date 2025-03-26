from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import subprocess

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 3, 30),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'drift_monitoring_and_retraining',
    default_args=default_args,
    schedule_interval='@daily',  # Run daily
    catchup=False,
)

def run_drift_detection():
    subprocess.run(["python", "monitoring/drift_detection.py"], check=True)

detect_drift = PythonOperator(
    task_id='detect_data_drift',
    python_callable=run_drift_detection,
    dag=dag,
)

detect_drift

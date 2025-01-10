PROJECT_NAME = "Telco Churn Classifier Dashboard"

WORKSACE_PATH = "/home/ubuntu/model-monitoring-assignment/monitoring-dashboard/customer_churn_model/evidently-data"

TRAINING_DATA_PATH = "./original-base-model-training-artifacts/training/training_data_with_predictions.csv"

NEW_DATA_PATH = '/home/ubuntu/model-monitoring-assignment/monitoring-dashboard/customer_churn_model/new-data-incoming.csv'

AIRFLOW_API_URL = 'http://localhost:8080/api/v1/dags/churn-classifier-model-retraining/dagRuns'
AIRFLOW_USERNAME = 'airflow'
AIRFLOW_PASSWORD = 'airflow'

BUCKET_NAME = 'churn-classifier-data'
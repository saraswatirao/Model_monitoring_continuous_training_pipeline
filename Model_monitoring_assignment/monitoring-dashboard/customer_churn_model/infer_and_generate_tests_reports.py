import pandas as pd
import numpy as np
import joblib
import config as cfg
from evidently.ui.workspace import Workspace
from reports_tests_utils import get_report, get_test_suite

import requests
import argparse
from requests.auth import HTTPBasicAuth
import boto3
import os

aws_access_key_id = os.environ['AWS_ACCESS_KEY_ID']
aws_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']

s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

def trigger_dag(original_train_data, new_train_data):

    parameters = {
        "conf": {
            "original_train_data": original_train_data,
            "new_train_data": new_train_data,
            "bucket_name": cfg.BUCKET_NAME
        }
    }   

    # Send an authenticated HTTP POST request to trigger the DAG
    response = requests.post(cfg.AIRFLOW_API_URL, json=parameters, auth=HTTPBasicAuth(cfg.AIRFLOW_USERNAME, cfg.AIRFLOW_PASSWORD))

    # Check the response to see if the DAG was triggered successfully
    if response.status_code == 200:
        print("DAG has been triggered successfully.")
    else:
        print(f"Failed to trigger the DAG. Status code: {response.status_code}")
        print(response.text)


def get_reference_data():

    """
    Returns the reference data to find drift against. In this case, we are using the training data as reference.
    """

    df = pd.read_csv(cfg.TRAINING_DATA_PATH)
    df['Predicted_Churn'] = df['Churn']

    return df

def get_args():

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Define the named parameters you want to accept
    parser.add_argument('--new_data_file', type=str, help="Name of the new incoming data file")

    # Parse the command-line arguments
    args = parser.parse_args()

    return args

def get_test_summary(tests_results):

    # False for failed and True for passed
    test_summary = []

    for result in tests_results:
        if result['status'] == 'SUCCESS':
            test_summary.append(True)
        else:
            test_summary.append(False)

    return test_summary

if __name__ == '__main__':

    s3.download_file(cfg.BUCKET_NAME, 'models/model_latest.pkl', 'models/model_latest.pkl')
    s3.download_file(cfg.BUCKET_NAME, 'models/encoder_latest.pkl', 'models/encoder_latest.pkl')
    s3.download_file(cfg.BUCKET_NAME, 'models/imputer_latest.pkl', 'models/imputer_latest.pkl')

    # Load your pre-trained machine learning model
    model = joblib.load('models/model_latest.pkl')

    # Load the OneHotEncoder from a joblib file
    encoder = joblib.load('models/encoder_latest.pkl')

    # Load the Imputer from a joblib file
    imputer = joblib.load('models/imputer_latest.pkl')

    args = get_args()

    # Load the CSV file you want to make predictions on
    new_data = pd.read_csv(args.new_data_file)

    # Preprocess the new data
    categorical_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                        'PaperlessBilling', 'PaymentMethod']

    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

    # Vectorize the categorical columns using the loaded encoder
    new_data_categorical = encoder.transform(new_data[categorical_cols])
    # Combine categorical and numerical data
    new_data_numerical = imputer.transform(new_data[numerical_cols])  # Extract numerical columns
    new_data_processed = np.concatenate([new_data_categorical, new_data_numerical], axis=1)

    # Make predictions using the loaded model
    predictions = model.predict(new_data_processed)

    # Add the predictions to the DataFrame
    new_data['Predicted_Churn'] = predictions

    # Returns the original created workspace if it exists already
    ws = Workspace.create(cfg.WORKSACE_PATH)

    existing_projects = ws.search_project(project_name=cfg.PROJECT_NAME)
    project = existing_projects[0]

    reference_data = get_reference_data()

    report = get_report(reference_data, new_data)
    ws.add_report(project.id, report)

    test_suite = get_test_suite(reference_data, new_data)

    ws.add_test_suite(project.id, test_suite)

    # print(test_suite.as_dict())

    test_summary = get_test_summary(test_suite.as_dict()['tests'])

    print(test_summary)
    print(test_suite.as_dict()['tests'])

    print("Uploading new data and original training data to S3")

    # Upload the model and encoder files to S3
    s3.upload_file(cfg.TRAINING_DATA_PATH, cfg.BUCKET_NAME, 'train-raw.csv')
    s3.upload_file(args.new_data_file, cfg.BUCKET_NAME, args.new_data_file)

    if not all(test_summary):
        print("One or more tests failed. Triggering model retraining...")
        trigger_dag('train-raw.csv', args.new_data_file)

    else:
        print("All tests have passed. Not retraining the model.")


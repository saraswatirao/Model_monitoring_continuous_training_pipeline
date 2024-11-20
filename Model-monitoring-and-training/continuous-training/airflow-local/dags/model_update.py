from airflow.models import DAG
from datetime import datetime, timedelta
from airflow.operators.python_operator import PythonOperator
from airflow.sensors.s3_key_sensor import S3KeySensor
import boto3
import os
from sklearn.feature_extraction.text import CountVectorizer
from active_learning import least_confidence_sampling
from train import train_and_evaluate_model

access_key = os.environ['AWS_ACCESS_KEY_ID']
secret_key = os.environ['AWS_SECRET_ACCESS_KEY']

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2022, 2, 22),
    'email': ['nic@enye.tech'],
    'email_on_failure': False,
    'max_active_runs': 1,
    'email_on_retry': False,
    'retry_delay': timedelta(minutes=5)
}

dg = DAG('sentiment-classifier-model-retraining',
          schedule_interval='@daily',
          default_args=default_args,
          catchup=False
          )
s3_buckname = 'sentiment-classifier-data'
unlabeled_file_name = 'new-data-unlabeled.txt'
labeled_file_name = 'new-data-labeled.txt'

check_for_new_unlabeled_data = S3KeySensor(
    task_id='check_for_new_unlabeled_data',
    poke_interval=20,
    timeout=18000,
    soft_fail=False,
    retries=2,
    bucket_key=unlabeled_file_name,
    bucket_name=s3_buckname,
    aws_conn_id='sentiment-classifier-bucket-conn',
    dag=dg)


sample_datapoints_for_labeling = PythonOperator(
    task_id='sample_datapoints_for_labeling',
    python_callable=least_confidence_sampling,
    dag=dg)

check_if_labeled_file_has_arrived = S3KeySensor(
    task_id='check_if_labeled_file_has_arrived',
    poke_interval=20,
    timeout=18000,
    soft_fail=False,
    retries=2,
    bucket_key=labeled_file_name,
    bucket_name=s3_buckname,
    aws_conn_id='sentiment-classifier-bucket-conn',
    dag=dg)

retrain_model = PythonOperator(
    task_id='retrain_model',
    python_callable=train_and_evaluate_model,
    dag=dg)

check_for_new_unlabeled_data >> sample_datapoints_for_labeling >> check_if_labeled_file_has_arrived >> retrain_model
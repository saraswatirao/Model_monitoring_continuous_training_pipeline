from airflow.models import DAG
from datetime import datetime, timedelta
from airflow.operators.python_operator import PythonOperator
from train import train_and_evaluate_model

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

dg = DAG('churn-classifier-model-retraining',
          schedule_interval=None,
          default_args=default_args,
          catchup=False
          )


retrain_model = PythonOperator(
    task_id='retrain_model',
    python_callable=train_and_evaluate_model,
    # Pass the data and the bucket name to the callable
    op_kwargs={'original_train_data': '{{ dag_run.conf["original_train_data"] }}',  # Access parameters from DAG run conf
                   'new_train_data': '{{ dag_run.conf["new_train_data"] }}',
                   'bucket_name': '{{ dag_run.conf["bucket_name"] }}'
    },
    provide_context=True,
    dag=dg)

retrain_model
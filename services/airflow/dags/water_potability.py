"""All assignment stages, including actual Docker deployment, every five minutes."""
from datetime import datetime, timedelta, timezone
import os

from airflow.sdk import DAG
from airflow.providers.standard.operators.bash import BashOperator


with DAG(
    dag_id="water_potability_pipeline",
    description="Prepare water data, train with MLflow, build and deploy API and app",
    start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
    schedule=os.getenv("PIPELINE_SCHEDULE", "*/5 * * * *"),
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(minutes=30),
    default_args={"owner": "water-potability", "retries": 1,
                  "retry_delay": timedelta(seconds=30)},
    tags=["assignment", "mlflow", "deployment"],
) as dag:
    load_data = BashOperator(task_id="load_data", bash_command='"$ML_PYTHON" /opt/project/code/download.py')
    prepare_data = BashOperator(task_id="prepare_data", bash_command='"$ML_PYTHON" /opt/project/code/preprocessing.py')
    train_model = BashOperator(
        task_id="train_and_evaluate",
        bash_command='"$ML_PYTHON" /opt/project/code/train.py',
        env={"AIRFLOW_RUN_ID": "{{ run_id }}"}, append_env=True,
    )
    deploy = BashOperator(task_id="build_and_deploy", bash_command='"$ML_PYTHON" /opt/project/code/deploy.py')

    # equivalent to load_data >> prepare_data >> train_model >> deploy
    load_data.set_downstream(prepare_data)
    prepare_data.set_downstream(train_model)
    train_model.set_downstream(deploy)

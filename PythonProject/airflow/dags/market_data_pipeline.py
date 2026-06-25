from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator


PROJECT_DIR = "/opt/airflow/project"
SRC_DIR = f"{PROJECT_DIR}/PythonProject/src"
DBT_DIR = f"{PROJECT_DIR}/PythonProject/dbt_trading"
DBT_RUNTIME_FLAGS = "--target-path /tmp/dbt_target --log-path /tmp/dbt_logs"

DEFAULT_ENV = {
    "DBT_PROFILES_DIR": DBT_DIR,
}

default_args = {
    "owner": "data-engineering",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="market_data_elt_pipeline",
    description="Extract raw stock data, build dbt Silver/Gold models, and test data quality.",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["trading", "elt", "dbt"],
) as dag:
    extract_to_bronze = BashOperator(
        task_id="extract_to_bronze_raw_stock_prices",
        bash_command=f"cd {SRC_DIR} && python Data/update_amzn_data.py",
        env=DEFAULT_ENV,
        append_env=True,
        do_xcom_push=False,
    )

    build_silver_gold = BashOperator(
        task_id="dbt_run_silver_gold_models",
        bash_command=(
            f"cd {DBT_DIR} && "
            f"dbt run --project-dir {DBT_DIR} --profiles-dir {DBT_DIR} {DBT_RUNTIME_FLAGS}"
        ),
        env=DEFAULT_ENV,
        append_env=True,
        do_xcom_push=False,
    )

    test_models = BashOperator(
        task_id="dbt_test_data_quality",
        bash_command=(
            f"cd {DBT_DIR} && "
            f"dbt test --project-dir {DBT_DIR} --profiles-dir {DBT_DIR} {DBT_RUNTIME_FLAGS}"
        ),
        env=DEFAULT_ENV,
        append_env=True,
        do_xcom_push=False,
    )

    extract_to_bronze >> build_silver_gold >> test_models

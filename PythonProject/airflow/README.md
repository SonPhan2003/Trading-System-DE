# Airflow Orchestration

This folder contains the Airflow DAG and Docker image definition for orchestrating the local ELT pipeline.

Pipeline tasks:

```text
extract_to_bronze_raw_stock_prices
-> dbt_run_silver_gold_models
-> dbt_test_data_quality
```

The DAG expects database and AWS configuration from the container environment.
In the local Docker Compose setup, these values are loaded from `PythonProject/.env`:

```text
DB_HOST
DB_PORT
DB_USER
DB_PASSWORD
DB_NAME
ALPHAVANTAGE_API_KEY
STOCK_SYMBOLS
ALPHAVANTAGE_OUTPUT_SIZE
ALPHAVANTAGE_REQUEST_DELAY_SECONDS
AWS_REGION
S3_RAW_BUCKET
S3_RAW_PREFIX
```

The DAG itself only adds `DBT_PROFILES_DIR` for dbt. The Python ingestion and dbt profile read the rest from the environment.

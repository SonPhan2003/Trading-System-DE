# Airflow Orchestration

This folder contains the Airflow DAG and Docker image definition for orchestrating the local ELT pipeline.

Pipeline tasks:

```text
extract_to_bronze_raw_stock_prices
-> dbt_run_silver_gold_models
-> dbt_test_data_quality
```

The DAG expects Airflow Variables for database configuration:

```text
DB_PASSWORD
DB_HOST
DB_PORT
DB_USER
DB_NAME
```

Only `DB_PASSWORD` is required if the defaults match your local setup.

Alpha Vantage settings are read by the Python ingestion code from `PythonProject/.env`:

```text
ALPHAVANTAGE_API_KEY
STOCK_SYMBOLS
ALPHAVANTAGE_OUTPUT_SIZE
ALPHAVANTAGE_REQUEST_DELAY_SECONDS
```

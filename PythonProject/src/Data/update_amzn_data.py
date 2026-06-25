import sys, os
from time import sleep

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from alpha_vantage.timeseries import TimeSeries
from config import (
    get_alpha_vantage_api_key,
    get_alpha_vantage_output_size,
    get_alpha_vantage_request_delay_seconds,
    get_symbols,
)
from Data.Database import (
    create_pipeline_tables,
    create_raw_stock_price_table,
    finish_pipeline_run,
    insert_data_quality_checks,
    insert_raw_stock_prices,
    start_pipeline_run,
)
from Data.s3_storage import upload_raw_stock_prices
from Data.validation import validate_ohlcv_data, validation_passed

PIPELINE_NAME = "daily_stock_price_ingestion"


def normalize_alpha_vantage_daily(data):
    data.reset_index(inplace=True)
    data.columns = ["date", "open", "high", "low", "close", "volume"]
    return data


def ingest_symbol(ts, symbol, output_size):
    run_id = start_pipeline_run(PIPELINE_NAME, symbol)
    rows_extracted = 0
    rows_loaded_to_raw = 0

    try:
        print(f"\n=== Processing {symbol} ===")
        data, _ = ts.get_daily(symbol=symbol, outputsize=output_size)
        data = normalize_alpha_vantage_daily(data)
        rows_extracted = len(data)
        upload_raw_stock_prices(symbol, data, run_id)
        insert_raw_stock_prices(run_id, symbol, data)
        rows_loaded_to_raw = len(data)
        checks = validate_ohlcv_data(data, symbol)
        insert_data_quality_checks(run_id, symbol, checks)
        if not validation_passed(checks):
            raise ValueError(f"Data quality checks failed for {symbol}")

        finish_pipeline_run(run_id, "SUCCESS", rows_extracted, rows_loaded_to_raw)
        print("Raw ingestion complete. Run dbt to build Silver and Gold models.")
        print(data.head())
    except Exception as exc:
        finish_pipeline_run(run_id, "FAILED", rows_extracted, rows_loaded_to_raw, str(exc))
        raise


def main():
    create_pipeline_tables()
    create_raw_stock_price_table()
    ts = TimeSeries(key=get_alpha_vantage_api_key(), output_format="pandas")
    output_size = get_alpha_vantage_output_size()
    request_delay = get_alpha_vantage_request_delay_seconds()
    symbols = get_symbols()

    for index, symbol in enumerate(symbols):
        try:
            ingest_symbol(ts, symbol, output_size)
        except Exception as exc:
            print(f"Failed to ingest {symbol}: {exc}")

        if index < len(symbols) - 1:
            print(f"Waiting {request_delay} seconds before the next API request...")
            sleep(request_delay)


if __name__ == "__main__":
    main()

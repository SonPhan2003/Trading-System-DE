from datetime import datetime

import mysql.connector

from config import get_database_config


def get_connection():
    return mysql.connector.connect(**get_database_config())


def _execute_schema(sql):
    cnx = get_connection()
    cursor = cnx.cursor()
    cursor.execute(sql)
    cnx.commit()
    cursor.close()
    cnx.close()


def create_legacy_stock_table():
    _execute_schema("""
    CREATE TABLE IF NOT EXISTS daily_stock_data (
        symbol VARCHAR(10),
        date DATE,
        open FLOAT,
        high FLOAT,
        low FLOAT,
        close FLOAT,
        volume BIGINT,
        PRIMARY KEY (symbol, date)
    );
    """)
    print("Table 'daily_stock_data' created (or already exists).")


def create_symbol_table():
    _execute_schema("""
    CREATE TABLE IF NOT EXISTS dim_symbols (
        symbol VARCHAR(10) PRIMARY KEY,
        company_name VARCHAR(255),
        asset_type VARCHAR(50) DEFAULT 'Equity',
        exchange_name VARCHAR(100),
        currency VARCHAR(10) DEFAULT 'USD',
        is_active BOOLEAN DEFAULT TRUE,
        created_at DATETIME NOT NULL,
        updated_at DATETIME NOT NULL
    );
    """)
    print("Table 'dim_symbols' created (or already exists).")


def create_daily_price_fact_table():
    _execute_schema("""
    CREATE TABLE IF NOT EXISTS fact_daily_prices (
        symbol VARCHAR(10),
        date DATE,
        open DECIMAL(12, 4),
        high DECIMAL(12, 4),
        low DECIMAL(12, 4),
        close DECIMAL(12, 4),
        volume BIGINT,
        loaded_at DATETIME NOT NULL,
        PRIMARY KEY (symbol, date),
        FOREIGN KEY (symbol) REFERENCES dim_symbols(symbol)
    );
    """)
    print("Table 'fact_daily_prices' created (or already exists).")


def create_raw_stock_price_table():
    _execute_schema("""
    CREATE TABLE IF NOT EXISTS raw_stock_prices (
        raw_id BIGINT AUTO_INCREMENT PRIMARY KEY,
        pipeline_run_id BIGINT NOT NULL,
        source VARCHAR(50) NOT NULL,
        symbol VARCHAR(10) NOT NULL,
        source_date DATE NOT NULL,
        open DECIMAL(12, 4),
        high DECIMAL(12, 4),
        low DECIMAL(12, 4),
        close DECIMAL(12, 4),
        volume BIGINT,
        ingested_at DATETIME NOT NULL,
        UNIQUE KEY uq_raw_stock_price (source, symbol, source_date),
        FOREIGN KEY (pipeline_run_id) REFERENCES pipeline_runs(run_id)
    );
    """)
    print("Table 'raw_stock_prices' created (or already exists).")


def create_clean_daily_price_table():
    _execute_schema("""
    CREATE TABLE IF NOT EXISTS clean_daily_prices (
        symbol VARCHAR(10) NOT NULL,
        date DATE NOT NULL,
        open DECIMAL(12, 4) NOT NULL,
        high DECIMAL(12, 4) NOT NULL,
        low DECIMAL(12, 4) NOT NULL,
        close DECIMAL(12, 4) NOT NULL,
        volume BIGINT NOT NULL,
        pipeline_run_id BIGINT NOT NULL,
        validated_at DATETIME NOT NULL,
        PRIMARY KEY (symbol, date),
        FOREIGN KEY (symbol) REFERENCES dim_symbols(symbol),
        FOREIGN KEY (pipeline_run_id) REFERENCES pipeline_runs(run_id)
    );
    """)
    print("Table 'clean_daily_prices' created (or already exists).")


def create_market_data_tables():
    create_symbol_table()
    create_raw_stock_price_table()
    create_clean_daily_price_table()
    create_daily_price_fact_table()


def create_pipeline_tables():
    _execute_schema("""
    CREATE TABLE IF NOT EXISTS pipeline_runs (
        run_id BIGINT AUTO_INCREMENT PRIMARY KEY,
        pipeline_name VARCHAR(100) NOT NULL,
        symbol VARCHAR(10),
        status VARCHAR(20) NOT NULL,
        rows_extracted INT DEFAULT 0,
        rows_loaded INT DEFAULT 0,
        started_at DATETIME NOT NULL,
        completed_at DATETIME,
        error_message TEXT
    );
    """)
    _execute_schema("""
    CREATE TABLE IF NOT EXISTS data_quality_checks (
        check_id BIGINT AUTO_INCREMENT PRIMARY KEY,
        run_id BIGINT NOT NULL,
        symbol VARCHAR(10) NOT NULL,
        check_name VARCHAR(100) NOT NULL,
        passed BOOLEAN NOT NULL,
        failed_rows INT DEFAULT 0,
        details TEXT,
        checked_at DATETIME NOT NULL,
        FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id)
    );
    """)
    print("Pipeline logging tables created (or already exist).")


def start_pipeline_run(pipeline_name, symbol):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO pipeline_runs (pipeline_name, symbol, status, started_at)
        VALUES (%s, %s, %s, %s)
        """,
        (pipeline_name, symbol, "RUNNING", datetime.utcnow()),
    )
    conn.commit()
    run_id = cursor.lastrowid
    cursor.close()
    conn.close()
    return run_id


def finish_pipeline_run(run_id, status, rows_extracted=0, rows_loaded=0, error_message=None):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE pipeline_runs
        SET status = %s,
            rows_extracted = %s,
            rows_loaded = %s,
            completed_at = %s,
            error_message = %s
        WHERE run_id = %s
        """,
        (status, rows_extracted, rows_loaded, datetime.utcnow(), error_message, run_id),
    )
    conn.commit()
    cursor.close()
    conn.close()


def insert_data_quality_checks(run_id, symbol, checks):
    conn = get_connection()
    cursor = conn.cursor()
    for check in checks:
        cursor.execute(
            """
            INSERT INTO data_quality_checks
                (run_id, symbol, check_name, passed, failed_rows, details, checked_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                run_id,
                symbol,
                check["check_name"],
                check["passed"],
                check["failed_rows"],
                check.get("details", ""),
                datetime.utcnow(),
            ),
        )
    conn.commit()
    cursor.close()
    conn.close()


def upsert_symbol(symbol, company_name=None, asset_type="Equity", exchange_name=None, currency="USD"):
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.utcnow()
    cursor.execute(
        """
        INSERT INTO dim_symbols
            (symbol, company_name, asset_type, exchange_name, currency, is_active, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, TRUE, %s, %s)
        ON DUPLICATE KEY UPDATE
            company_name = COALESCE(VALUES(company_name), company_name),
            asset_type = VALUES(asset_type),
            exchange_name = COALESCE(VALUES(exchange_name), exchange_name),
            currency = VALUES(currency),
            is_active = TRUE,
            updated_at = VALUES(updated_at)
        """,
        (symbol, company_name, asset_type, exchange_name, currency, now, now),
    )
    conn.commit()
    cursor.close()
    conn.close()


def insert_raw_stock_prices(run_id, symbol, df, source="alpha_vantage"):
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.utcnow()

    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO raw_stock_prices
                (pipeline_run_id, source, symbol, source_date, open, high, low, close, volume, ingested_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                pipeline_run_id = VALUES(pipeline_run_id),
                open = VALUES(open),
                high = VALUES(high),
                low = VALUES(low),
                close = VALUES(close),
                volume = VALUES(volume),
                ingested_at = VALUES(ingested_at)
        """, (
            run_id,
            source,
            symbol,
            row["date"].date(),
            float(row["open"]),
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
            int(row["volume"]),
            now,
        ))

    conn.commit()
    cursor.close()
    conn.close()
    print(f"Inserted {len(df)} raw rows for {symbol} into raw_stock_prices.")


def get_latest_date(symbol):
    cnx = get_connection()
    cursor = cnx.cursor()
    cursor.execute("SELECT MAX(date) FROM fact_daily_prices WHERE symbol = %s;", (symbol,))
    result = cursor.fetchone()[0]
    cursor.close()
    cnx.close()
    return result


def transform_raw_to_clean(run_id, symbol, source="alpha_vantage", latest_date=None):
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.utcnow()
    cursor.execute("""
        INSERT INTO clean_daily_prices
            (symbol, date, open, high, low, close, volume, pipeline_run_id, validated_at)
        SELECT
            symbol,
            source_date,
            open,
            high,
            low,
            close,
            volume,
            pipeline_run_id,
            %s
        FROM raw_stock_prices
        WHERE pipeline_run_id = %s
          AND source = %s
          AND symbol = %s
          AND (%s IS NULL OR source_date > %s)
          AND source_date IS NOT NULL
          AND open IS NOT NULL
          AND high IS NOT NULL
          AND low IS NOT NULL
          AND close IS NOT NULL
          AND volume IS NOT NULL
          AND open > 0
          AND high > 0
          AND low > 0
          AND close > 0
          AND high >= low
          AND volume >= 0
        ON DUPLICATE KEY UPDATE
            open = VALUES(open),
            high = VALUES(high),
            low = VALUES(low),
            close = VALUES(close),
            volume = VALUES(volume),
            pipeline_run_id = VALUES(pipeline_run_id),
            validated_at = VALUES(validated_at)
    """, (now, run_id, source, symbol, latest_date, latest_date))
    transformed_rows = cursor.rowcount
    conn.commit()
    cursor.close()
    conn.close()
    print(f"Transformed raw rows into clean_daily_prices for {symbol}: {transformed_rows}")
    return transformed_rows


def load_clean_prices_to_fact(run_id, symbol):
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.utcnow()
    cursor.execute("""
        INSERT INTO fact_daily_prices
            (symbol, date, open, high, low, close, volume, loaded_at)
        SELECT
            symbol,
            date,
            open,
            high,
            low,
            close,
            volume,
            %s
        FROM clean_daily_prices
        WHERE pipeline_run_id = %s
          AND symbol = %s
        ON DUPLICATE KEY UPDATE
            open = VALUES(open),
            high = VALUES(high),
            low = VALUES(low),
            close = VALUES(close),
            volume = VALUES(volume),
            loaded_at = VALUES(loaded_at)
    """, (now, run_id, symbol))
    loaded_rows = cursor.rowcount
    conn.commit()
    cursor.close()
    conn.close()
    print(f"Loaded clean rows into fact_daily_prices for {symbol}: {loaded_rows}")
    return loaded_rows


def insert_daily_prices(symbol, df):
    conn = get_connection()
    cursor = conn.cursor()

    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO fact_daily_prices
                (symbol, date, open, high, low, close, volume, loaded_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                open = VALUES(open),
                high = VALUES(high),
                low = VALUES(low),
                close = VALUES(close),
                volume = VALUES(volume),
                loaded_at = VALUES(loaded_at)
        """, (
            symbol,
            row['date'].date(),
            float(row['open']),
            float(row['high']),
            float(row['low']),
            float(row['close']),
            int(row['volume']),
            datetime.utcnow(),
        ))

    conn.commit()
    cursor.close()
    conn.close()
    print(f"Inserted {len(df)} rows for {symbol} into fact_daily_prices.")


def legacy_stock_table_exists():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SHOW TABLES LIKE 'daily_stock_data'")
    exists = cursor.fetchone() is not None
    cursor.close()
    conn.close()
    return exists


def migrate_legacy_stock_data_to_fact():
    if not legacy_stock_table_exists():
        print("Legacy table 'daily_stock_data' not found. Skipping migration.")
        return 0

    run_id = start_pipeline_run("legacy_stock_data_migration", "ALL")
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.utcnow()

    try:
        cursor.execute("""
            INSERT INTO dim_symbols
                (symbol, company_name, asset_type, exchange_name, currency, is_active, created_at, updated_at)
            SELECT DISTINCT symbol, NULL, 'Equity', NULL, 'USD', TRUE, %s, %s
            FROM daily_stock_data
            ON DUPLICATE KEY UPDATE
                is_active = TRUE,
                updated_at = VALUES(updated_at)
        """, (now, now))

        cursor.execute("""
            INSERT INTO clean_daily_prices
                (symbol, date, open, high, low, close, volume, pipeline_run_id, validated_at)
            SELECT symbol, date, open, high, low, close, volume, %s, %s
            FROM daily_stock_data
            WHERE date IS NOT NULL
              AND open IS NOT NULL
              AND high IS NOT NULL
              AND low IS NOT NULL
              AND close IS NOT NULL
              AND volume IS NOT NULL
              AND open > 0
              AND high > 0
              AND low > 0
              AND close > 0
              AND high >= low
              AND volume >= 0
            ON DUPLICATE KEY UPDATE
                open = VALUES(open),
                high = VALUES(high),
                low = VALUES(low),
                close = VALUES(close),
                volume = VALUES(volume),
                pipeline_run_id = VALUES(pipeline_run_id),
                validated_at = VALUES(validated_at)
        """, (run_id, now))

        cursor.execute("""
            INSERT INTO fact_daily_prices
                (symbol, date, open, high, low, close, volume, loaded_at)
            SELECT symbol, date, open, high, low, close, volume, %s
            FROM clean_daily_prices
            WHERE pipeline_run_id = %s
            ON DUPLICATE KEY UPDATE
                open = VALUES(open),
                high = VALUES(high),
                low = VALUES(low),
                close = VALUES(close),
                volume = VALUES(volume),
                loaded_at = VALUES(loaded_at)
        """, (now, run_id))
        migrated_rows = cursor.rowcount

        conn.commit()
        finish_pipeline_run(run_id, "SUCCESS", rows_loaded=migrated_rows)
        print(f"Migrated legacy daily_stock_data rows through clean_daily_prices into fact_daily_prices: {migrated_rows}")
        return migrated_rows
    except Exception as exc:
        conn.rollback()
        finish_pipeline_run(run_id, "FAILED", error_message=str(exc))
        raise
    finally:
        cursor.close()
        conn.close()


def create_stock_table():
    create_legacy_stock_table()


def insert_stock_data(symbol, df):
    insert_daily_prices(symbol, df)

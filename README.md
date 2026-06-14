# Trading System - Data Engineering and Strategy Analysis Framework

This project ingests daily stock market data from Alpha Vantage, stores it in
MySQL, validates incoming OHLCV records, logs pipeline execution metadata, and
serves the processed data to a Streamlit strategy analysis dashboard.

The project started as a thesis trading strategy evaluation framework. It now
also includes Data Engineering components such as environment-based
configuration, incremental loading, data quality checks, and pipeline run
logging.

## 0. Quick Start (for graders)

These steps are the fastest way to run the project. All commands assume you are
in the repository root.

**A) Start MySQL (recommended via Docker):**
```bash
docker run -d --name trading-mysql \
  -e MYSQL_ROOT_PASSWORD=YOUR_PASSWORD_HERE \
  -e MYSQL_DATABASE=Logging \
  -p 3306:3306 \
  mysql:9.2.0

# check
docker ps
```

**B) Install Python dependencies:**
```bash
cd PythonProject
pip install -r requirements.txt
```

**C) Configure environment variables:**
```bash
copy .env.example .env
copy PythonProject\.env.example PythonProject\.env
```

Edit `PythonProject/.env` for the Python ingestion app. If you keep a local
Docker Compose file, keep it private and use the same database password there.

Recommended free-tier Alpha Vantage settings:
```env
STOCK_SYMBOLS=AMZN,AAPL,MSFT
ALPHAVANTAGE_OUTPUT_SIZE=compact
ALPHAVANTAGE_REQUEST_DELAY_SECONDS=15
```

**D) Load or update raw stock data:**
```bash
cd src
python Data/update_amzn_data.py
```

**E) Build dbt Silver/Gold models:**
```bash
cd ..\..
docker build -t trading-dbt PythonProject\dbt_trading

docker run --rm `
  -v "${PWD}\PythonProject\dbt_trading:/usr/app" `
  -w /usr/app `
  -e DB_HOST=host.docker.internal `
  -e DB_PORT=3306 `
  -e DB_USER=root `
  -e DB_PASSWORD=YOUR_PASSWORD_HERE `
  -e DB_NAME=Logging `
  -e DBT_PROFILES_DIR=/usr/app `
  trading-dbt dbt run

docker run --rm `
  -v "${PWD}\PythonProject\dbt_trading:/usr/app" `
  -w /usr/app `
  -e DB_HOST=host.docker.internal `
  -e DB_PORT=3306 `
  -e DB_USER=root `
  -e DB_PASSWORD=YOUR_PASSWORD_HERE `
  -e DB_NAME=Logging `
  -e DBT_PROFILES_DIR=/usr/app `
  trading-dbt dbt test
```

**F) Run the application:**
```bash
cd PythonProject\src
streamlit run compare_strategies.py
```

---

## Table of Contents
1. [Environment Requirements](#1-environment-requirements)
2. [Setup Steps](#2-setup-steps)
3. [How to Retrain the Model](#3-how-to-retrain-the-model)
4. [How to Run the Program](#4-how-to-run-the-program)
5. [Data Engineering Components](#5-data-engineering-components)

---

## 1. Environment Requirements

### 1.1. System Requirements

**Operating System:**
- Windows 10/11, macOS, or Linux

**Software Requirements:**
- **Python**: Version 3.8 or higher
- **MySQL Server**: Version 8.0 or higher (or Docker to run MySQL)
- **pip**: Python package manager (usually comes with Python)

**Hardware Requirements:**
- Minimum 4GB RAM (8GB recommended)
- At least 2GB free disk space
- Internet connection (for downloading dependencies and stock data)

### 1.2. Python Libraries Required

Install the following Python packages:
- `streamlit` - Web interface framework
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `torch` - PyTorch for LSTM model
- `mysql-connector-python` - MySQL database connector
- `scikit-learn` - Machine learning utilities
- `alpha-vantage` - Stock data API (for data loading)

### 1.3. Database Configuration

The Python ingestion app reads database and API settings from `PythonProject/.env`.

Required `PythonProject/.env` values:
```env
DB_USER=root
DB_PASSWORD=YOUR_PASSWORD_HERE
DB_HOST=127.0.0.1
DB_NAME=Logging
DB_PORT=3306
ALPHAVANTAGE_API_KEY=YOUR_KEY_HERE
ALPHAVANTAGE_OUTPUT_SIZE=compact
ALPHAVANTAGE_REQUEST_DELAY_SECONDS=15
STOCK_SYMBOLS=AMZN,AAPL,MSFT
```

Keep real secrets in `.env` files. Commit only `.env.example` files.

---

## 2. Setup Steps

### Step 1: Install MySQL Database

**Recommended: Use Docker** (easiest setup)

```bash
# Run MySQL in Docker container
docker run -d \
  --name trading-mysql \
  -e MYSQL_ROOT_PASSWORD=YOUR_PASSWORD_HERE \
  -e MYSQL_DATABASE=Logging \
  -p 3306:3306 \
  mysql:9.2.0

# Verify container is running
docker ps
```

**Note**: Ensure port 3306 is not already in use by another MySQL instance.

**Alternative: Install MySQL Directly**

If you prefer not to use Docker:
1. Download and install MySQL from https://dev.mysql.com/downloads/mysql/
2. Start MySQL service: `net start MySQL80` (Windows) or `sudo systemctl start mysql` (Linux/macOS)
3. Set a local root password and create database `Logging`
4. Or update `PythonProject/.env` to match your MySQL configuration

### Step 2: Install Python Dependencies

1. **Navigate to project directory:**
   ```bash
   cd PythonProject
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

### Step 3: Configure Environment Variables

1. **Create your local environment file:**
   ```bash
   copy PythonProject\.env.example PythonProject\.env
   ```

2. **Edit `PythonProject/.env` with your local values:**
   - Set `DB_PASSWORD` to match your MySQL container.
   - Set `ALPHAVANTAGE_API_KEY` to your Alpha Vantage key.
   - Keep `ALPHAVANTAGE_OUTPUT_SIZE=compact` for the free API tier.
   - Keep a small `STOCK_SYMBOLS` list when using the free API tier.

### Step 4: Load Stock Data into Database

1. **Navigate to source directory:**
   ```bash
   cd PythonProject/src
   ```

2. **Run the data loading script:**
   ```bash
   python Data/update_amzn_data.py
   ```

   This script will:
   - Automatically create the `raw_stock_prices` Bronze table
   - Automatically create `pipeline_runs` and `data_quality_checks`
   - Download stock data for symbols configured in `STOCK_SYMBOLS`
   - Persist source API data in the Bronze raw table
   - Log basic data quality check results
   - Log each ingestion run with status, row counts, timestamps, and errors

   dbt then transforms raw data into Silver and Gold tables.

   **Note**: The script uses Alpha Vantage API. Free API keys can hit rate
   limits, so use a small symbol list and keep
   `ALPHAVANTAGE_REQUEST_DELAY_SECONDS` configured.

3. **Verify data was loaded:**
   ```python
   # Test in Python
   from Data.Database import get_connection
   import pandas as pd
   
   conn = get_connection()
   df = pd.read_sql("SELECT COUNT(*) as count FROM raw_stock_prices", conn)
   print(f"Total records: {df['count'].iloc[0]}")
   conn.close()
   ```

### Step 5: Verify Installation

Test that everything is set up correctly:

```python
# Test database connection
from Data.Database import get_connection
conn = get_connection()
print("Database connection successful!")
conn.close()

# Test data loading
from Data.Loader import load_stock_data
df = load_stock_data("AMZN")
print(f"Data loaded successfully! Records: {len(df)}")
```

---

## 3. How to Retrain the Model

The LSTM model is pre-trained and saved as `lstm_AMZN_2class.pt`. However, if you want to retrain the model with new data or different parameters, follow these steps:

### 3.1. Understanding the Model

The LSTM model is a binary classifier that predicts:
- **Class 0**: Stock price will decrease (return < -threshold)
- **Class 1**: Stock price will increase (return > threshold)

### 3.2. Training Process

1. **Open Python in the project directory:**
   ```bash
   cd PythonProject/src
   python
   ```

2. **Import required modules:**
   ```python
   from Data.Loader import load_stock_data
   from strategies.LSTM_strategy import lstm_strategy
   ```

3. **Load stock data:**
   ```python
   # Load data for the symbol you want to train on
   df = load_stock_data("AMZN")
   ```

4. **Run training:**
   ```python
   # Train the LSTM model
   trades, model_path = lstm_strategy(
       symbol="AMZN",           # Stock symbol
       lookback=30,             # Number of days to look back
       batch_size=64,           # Batch size for training
       epochs=15,               # Number of training epochs
       lr=1e-3,                 # Learning rate
       threshold=0.001          # Threshold for creating labels (0.1%)
   )
   ```

   **Training Parameters:**
   - `lookback`: Number of historical days used for prediction (default: 30)
   - `batch_size`: Number of samples per training batch (default: 64)
   - `epochs`: Number of complete passes through the training data (default: 15)
   - `lr`: Learning rate for the optimizer (default: 0.001)
   - `threshold`: Minimum return percentage to create a positive label (default: 0.001 = 0.1%)

5. **Model will be saved:**
   - Location: `PythonProject/src/strategies/lstm_AMZN_2class.pt`
   - This file contains the trained model weights
   - The model can be loaded and used without retraining

### 3.3. Training Details

**Data Preparation:**
- The model uses 16 technical indicators as features:
  - Moving Averages: SMA_5, SMA_10, EMA_10
  - RSI (14 periods)
  - MACD (12, 26, 9)
  - ATR (14 periods)
  - Bollinger Bands (20 periods, 2 standard deviations)
  - OHLCV (Open, High, Low, Close, Volume)

**Data Processing:**
- Features are normalized using StandardScaler
- Data is split: 70% for training, 30% for testing
- Sliding windows of 30 days are created for sequence prediction

**Model Architecture:**
- LSTM layer: 16 input features → 64 hidden units
- Fully connected layer: 64 → 2 classes (binary classification)
- Activation: Softmax (from logits)

**Training Time:**
- Typically takes 5-15 minutes depending on dataset size
- Requires at least a few years of historical data for good performance

### 3.4. Important Notes

- **Pre-trained model included**: The file `lstm_AMZN_2class.pt` is already included in the project. You don't need to retrain unless you want to:
  - Use different parameters
  - Train on different stock symbols
  - Retrain with updated data

- **Overfitting**: If you have limited data, the model may overfit. Consider:
  - Reducing the number of epochs
  - Adding regularization
  - Using more training data

---

## 4. How to Run the Program

### 4.1. Prerequisites Check

Before running, ensure:
- MySQL is running (check with `docker ps` or `net start MySQL80`)
- Database has stock data loaded (run `update_amzn_data.py` if needed)
- All Python dependencies are installed
- Pre-trained model file exists: `strategies/lstm_AMZN_2class.pt`

### 4.2. Start the Streamlit Application

1. **Navigate to source directory:**
   ```bash
   cd PythonProject/src
   ```

2. **Run Streamlit:**
   ```bash
   streamlit run compare_strategies.py
   ```

3. **Access the application:**
   - The app will automatically open in your default browser
   - URL: `http://localhost:8501`
   - If it doesn't open automatically, copy the URL from the terminal

### 4.3. Using the Application

#### 4.3.1. Configure Inputs (Sidebar)

Before running strategies, configure:
- **Primary symbol**: Main stock symbol to analyze (default: AMZN)
- **Pair symbol**: Second symbol for Pairs Cointegration strategy (default: MSFT)
- **Start date**: Beginning date for backtesting
- **End date**: Ending date for backtesting
- **Mode**: 
  - `Run single` - Run one strategy at a time
  - `Compare all` - Compare all 5 strategies simultaneously

#### 4.3.2. Run Single Strategy

1. Select `Run single` mode
2. Choose a strategy from the dropdown:
   - Mean Reverting
   - Bollinger + RSI
   - Pairs Cointegration
   - LSTM
   - Random Forest
3. Click the `Run` button
4. View results:
   - Total number of trades generated
   - Recent trades table
   - Performance metrics (Sharpe Ratio, Max Drawdown, Total Return, etc.)
   - Equity curve comparison with SPY benchmark
   - Drawdown analysis chart
   - Rolling volatility chart
   - Monthly returns chart

#### 4.3.3. Compare All Strategies

1. Select `Compare all` mode
2. Click the `Run` button
3. Wait for all strategies to complete (may take 1-2 minutes)
4. View comprehensive comparison:
   - Summary table with all metrics for each strategy
   - Equity curves comparison chart
   - Drawdown comparison chart
   - Sharpe Ratio comparison bar chart
   - Volatility comparison bar chart
   - Max Drawdown comparison bar chart
   - Total Return comparison bar chart

#### 4.3.4. Understanding the Results

**Key Metrics Explained:**
- **Total Return**: Overall profit/loss percentage over the period
- **Sharpe Ratio**: Risk-adjusted return (higher is better, >1.0 is good)
- **Max Drawdown**: Maximum loss from peak (lower/less negative is better)
- **Volatility**: Annualized standard deviation of returns (lower is generally better)
- **Win Rate**: Percentage of profitable trades
- **Sortino Ratio**: Similar to Sharpe but only considers downside risk

**Benchmark Comparison:**
- All strategies are compared against SPY (S&P 500 ETF)
- A strategy outperforms if its equity curve is above the benchmark line
- Lower drawdown than benchmark indicates better risk management

---

## 5. Data Engineering Components

### 5.1. Medallion-Style ELT Model

The market data storage layer uses a Medallion-style ELT design:

```text
Bronze: raw_stock_prices
Silver: dbt_clean_daily_prices
Gold: dbt_dim_symbols, dbt_fact_daily_prices
```

`raw_stock_prices` stores source API records as they were ingested, including
the source name, symbol, source date, ingestion timestamp, and pipeline run id.

`dbt_clean_daily_prices` stores validated OHLCV rows produced from the raw table
using SQL transformation rules.

`dbt_dim_symbols` stores descriptive symbol-level data, such as the ticker,
asset type, currency, and active status.

`dbt_fact_daily_prices` stores measurable daily OHLCV observations by symbol and
date. The primary key is `(symbol, date)`, which prevents duplicate daily
records and supports time-series analytics.

This separates entity metadata from daily market measurements, making the
database easier to extend for analytics and reporting.

Current analytical schema:
```text
raw_stock_prices
- raw_id
- pipeline_run_id
- source
- symbol
- source_date
- open
- high
- low
- close
- volume
- ingested_at

dbt_clean_daily_prices
- symbol
- date
- open
- high
- low
- close
- volume
- pipeline_run_id
- validated_at

dbt_dim_symbols
- symbol
- company_name
- asset_type
- exchange_name
- currency
- is_active
- created_at
- updated_at

dbt_fact_daily_prices
- symbol
- date
- open
- high
- low
- close
- volume
- loaded_at
```

### 5.2. Ingestion Pipeline

The ingestion script is located at:
```text
PythonProject/src/Data/update_amzn_data.py
```

It extracts daily OHLCV data from Alpha Vantage, applies data quality checks,
loads new records into MySQL, and logs each pipeline run.

Pipeline flow:
```text
Python ingestion: Alpha Vantage API -> raw_stock_prices
dbt run: raw_stock_prices -> dbt_clean_daily_prices
dbt run: dbt_clean_daily_prices -> dbt_dim_symbols / dbt_fact_daily_prices
-> Streamlit strategy analysis
```

### 5.3. Incremental Loading

The pipeline checks the latest stored date for each symbol before inserting
data. This avoids reloading the same records every time the pipeline runs.

### 5.4. Data Quality Checks

The validation logic is located at:
```text
PythonProject/src/Data/validation.py
```

Current checks include:
- Required OHLCV columns are present
- No null OHLCV values
- No duplicate dates per API response
- Prices are positive
- High price is greater than or equal to low price
- Volume is non-negative

### 5.5. Pipeline Observability

The ingestion process writes metadata to:
```text
pipeline_runs
data_quality_checks
```

These tables track pipeline status, row counts, timestamps, errors, and data
quality results.

### 5.6. dbt Transformation Layer

The project also includes a dbt project at:

```text
PythonProject/dbt_trading
```

The dbt setup runs in Docker. It builds separate dbt-managed models first,
using a `dbt_` prefix so the existing Streamlit workflow remains safe while
the dbt layer is tested.

Current dbt models:
```text
dbt_clean_daily_prices
dbt_dim_symbols
dbt_fact_daily_prices
```

dbt flow:
```text
raw_stock_prices
-> dbt_clean_daily_prices
-> dbt_dim_symbols / dbt_fact_daily_prices
```

Run dbt:
```powershell
docker build -t trading-dbt PythonProject\dbt_trading

docker run --rm `
  -v "${PWD}\PythonProject\dbt_trading:/usr/app" `
  -w /usr/app `
  -e DB_HOST=host.docker.internal `
  -e DB_PORT=3306 `
  -e DB_USER=root `
  -e DB_PASSWORD=YOUR_PASSWORD_HERE `
  -e DB_NAME=Logging `
  -e DBT_PROFILES_DIR=/usr/app `
  trading-dbt dbt debug

docker run --rm `
  -v "${PWD}\PythonProject\dbt_trading:/usr/app" `
  -w /usr/app `
  -e DB_HOST=host.docker.internal `
  -e DB_PORT=3306 `
  -e DB_USER=root `
  -e DB_PASSWORD=YOUR_PASSWORD_HERE `
  -e DB_NAME=Logging `
  -e DBT_PROFILES_DIR=/usr/app `
  trading-dbt dbt run

docker run --rm `
  -v "${PWD}\PythonProject\dbt_trading:/usr/app" `
  -w /usr/app `
  -e DB_HOST=host.docker.internal `
  -e DB_PORT=3306 `
  -e DB_USER=root `
  -e DB_PASSWORD=YOUR_PASSWORD_HERE `
  -e DB_NAME=Logging `
  -e DBT_PROFILES_DIR=/usr/app `
  trading-dbt dbt test
```

The dbt layer is intentionally separate from the current production tables.
After verifying it, the project can switch Streamlit to the dbt-managed Gold
tables or rename the dbt models to produce the final tables directly.

---

## Troubleshooting

### Database Connection Error

**Error**: `Can't connect to MySQL server`

**Solutions:**
- Check MySQL is running: `net start MySQL80` (Windows) or `sudo systemctl status mysql` (Linux/macOS)
- If using Docker, check with `docker ps` or start it with `docker start trading-mysql`
- Verify credentials in `PythonProject/.env` match your MySQL setup
- Ensure port 3306 is not blocked by firewall
- Test connection: `mysql -u root -p -h 127.0.0.1 Logging`

### Alpha Vantage Rate Limit Error

**Error**: `Please consider spreading out your free API requests`

**Solutions:**
- Reduce `STOCK_SYMBOLS` in `PythonProject/.env`
- Increase `ALPHAVANTAGE_REQUEST_DELAY_SECONDS`
- Keep `ALPHAVANTAGE_OUTPUT_SIZE=compact` for free API keys

### Model File Not Found

**Error**: `No saved model found for AMZN`

**Solutions:**
- Verify file exists: `PythonProject/src/strategies/lstm_AMZN_2class.pt`
- If missing, retrain the model (see Section 3)

### No Data Available

**Error**: `No data for symbol in selected range`

**Solutions:**
- Check database has data: `SELECT COUNT(*) FROM dbt_fact_daily_prices WHERE symbol='AMZN';`
- Run data loading script: `python Data/update_amzn_data.py`
- Run dbt: `dbt run` and `dbt test` through the Docker commands above
- Verify date range has available data

### Import Errors

**Error**: `ImportError: No module named 'xxx'`

**Solutions:**
- Install missing package: `pip install xxx`
- Install all dependencies: `pip install -r requirements.txt`
- Verify Python version: `python --version` (should be 3.8+)

---

## Quick Reference

**Start MySQL:**
```bash
# Windows
net start MySQL80

# Linux/macOS
sudo systemctl start mysql

# Docker
docker start trading-mysql
```

**Load Data:**
```bash
cd PythonProject/src
python Data/update_amzn_data.py
```

**Run Application:**
```bash
cd PythonProject/src
streamlit run compare_strategies.py
```

**Retrain Model:**
```python
from strategies.LSTM_strategy import lstm_strategy
trades, model_path = lstm_strategy("AMZN", lookback=30, batch_size=64, epochs=15, lr=1e-3, threshold=0.001)
```

---

**End of Installation Guide**

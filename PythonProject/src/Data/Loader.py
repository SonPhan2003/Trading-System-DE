import pandas as pd
from Data.Database import get_connection

def load_stock_data(symbol):
    """
    Loads historical data for a given stock symbol from the dbt Gold fact table.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT date, open, high, low, close, volume
        FROM dbt_fact_daily_prices
        WHERE symbol = %s
        ORDER BY date
        """,
        (symbol,)
    )
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(rows, columns=columns)
    df["date"] = pd.to_datetime(df["date"])
    numeric_columns = ["open", "high", "low", "close", "volume"]
    df[numeric_columns] = df[numeric_columns].astype(float)
    df.set_index("date", inplace=True)
    cursor.close()
    conn.close()
    return df

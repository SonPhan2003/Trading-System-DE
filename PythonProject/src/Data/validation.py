import pandas as pd


REQUIRED_PRICE_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def validate_ohlcv_data(df: pd.DataFrame, symbol: str) -> list[dict]:
    checks = []

    def add_check(name: str, passed: bool, failed_rows: int = 0, details: str = "") -> None:
        checks.append(
            {
                "check_name": name,
                "passed": bool(passed),
                "failed_rows": int(failed_rows),
                "details": details,
            }
        )

    missing_columns = [column for column in REQUIRED_PRICE_COLUMNS if column not in df.columns]
    add_check(
        "required_columns_present",
        not missing_columns,
        len(missing_columns),
        f"Missing columns for {symbol}: {', '.join(missing_columns)}" if missing_columns else "",
    )

    if missing_columns:
        return checks

    null_rows = int(df[REQUIRED_PRICE_COLUMNS].isna().any(axis=1).sum())
    add_check("no_null_ohlcv_values", null_rows == 0, null_rows)

    duplicate_rows = int(df.duplicated(subset=["date"]).sum())
    add_check("no_duplicate_dates", duplicate_rows == 0, duplicate_rows)

    invalid_price_rows = int(
        (
            (df["open"] <= 0)
            | (df["high"] <= 0)
            | (df["low"] <= 0)
            | (df["close"] <= 0)
        ).sum()
    )
    add_check("positive_prices", invalid_price_rows == 0, invalid_price_rows)

    invalid_high_low_rows = int((df["high"] < df["low"]).sum())
    add_check("high_greater_or_equal_low", invalid_high_low_rows == 0, invalid_high_low_rows)

    invalid_volume_rows = int((df["volume"] < 0).sum())
    add_check("non_negative_volume", invalid_volume_rows == 0, invalid_volume_rows)

    return checks


def validation_passed(checks: list[dict]) -> bool:
    return all(check["passed"] for check in checks)

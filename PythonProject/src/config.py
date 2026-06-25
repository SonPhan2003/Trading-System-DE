from pathlib import Path
from typing import Optional
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"


def _load_env_file(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_env_file()


def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def get_optional_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def get_database_config() -> dict:
    return {
        "user": get_required_env("DB_USER"),
        "password": get_required_env("DB_PASSWORD"),
        "host": os.getenv("DB_HOST", "127.0.0.1"),
        "database": os.getenv("DB_NAME", "Logging"),
        "port": int(os.getenv("DB_PORT", "3306")),
    }


def get_alpha_vantage_api_key() -> str:
    return get_required_env("ALPHAVANTAGE_API_KEY")


def get_symbols() -> list[str]:
    raw_symbols = os.getenv("STOCK_SYMBOLS", "AMZN,MSFT,AAPL,JPM,BAC,XOM,CVX,SPY")
    return [symbol.strip().upper() for symbol in raw_symbols.split(",") if symbol.strip()]


def get_alpha_vantage_output_size() -> str:
    output_size = os.getenv("ALPHAVANTAGE_OUTPUT_SIZE", "compact").strip().lower()
    if output_size not in {"compact", "full"}:
        raise RuntimeError("ALPHAVANTAGE_OUTPUT_SIZE must be either 'compact' or 'full'")
    return output_size


def get_alpha_vantage_request_delay_seconds() -> float:
    return float(os.getenv("ALPHAVANTAGE_REQUEST_DELAY_SECONDS", "15"))


def get_aws_region() -> str:
    return os.getenv("AWS_REGION", "ap-southeast-1")


def get_s3_raw_bucket() -> Optional[str]:
    return get_optional_env("S3_RAW_BUCKET")


def get_s3_raw_prefix() -> str:
    return os.getenv("S3_RAW_PREFIX", "alpha_vantage/daily_prices").strip().strip("/")

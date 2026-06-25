from __future__ import annotations

from datetime import datetime, timezone
from io import StringIO

import boto3
import pandas as pd

from config import get_aws_region, get_s3_raw_bucket, get_s3_raw_prefix


def upload_raw_stock_prices(symbol: str, data: pd.DataFrame, run_id: int) -> str | None:
    bucket = get_s3_raw_bucket()
    if not bucket:
        print("S3 raw bucket is not configured. Skipping S3 raw upload.")
        return None

    extracted_at = datetime.now(timezone.utc)
    date_part = extracted_at.strftime("%Y-%m-%d")
    timestamp_part = extracted_at.strftime("%Y%m%dT%H%M%SZ")
    prefix = get_s3_raw_prefix()
    key = (
        f"{prefix}/symbol={symbol}/extract_date={date_part}/"
        f"raw_daily_prices_{symbol}_{timestamp_part}_run_{run_id}.csv"
    )

    buffer = StringIO()
    data.to_csv(buffer, index=False)

    s3_client = boto3.client("s3", region_name=get_aws_region())
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=buffer.getvalue().encode("utf-8"),
        ContentType="text/csv",
        Metadata={
            "source": "alpha_vantage",
            "symbol": symbol,
            "pipeline_run_id": str(run_id),
            "extracted_at_utc": extracted_at.isoformat(),
        },
    )

    s3_uri = f"s3://{bucket}/{key}"
    print(f"Uploaded raw extract to {s3_uri}")
    return s3_uri

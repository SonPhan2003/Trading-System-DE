{{ config(alias='dbt_clean_daily_prices') }}

select
    symbol,
    source_date as date,
    cast(open as decimal(12, 4)) as open,
    cast(high as decimal(12, 4)) as high,
    cast(low as decimal(12, 4)) as low,
    cast(close as decimal(12, 4)) as close,
    cast(volume as signed) as volume,
    pipeline_run_id,
    current_timestamp as validated_at
from {{ source('trading_source', 'raw_stock_prices') }}
where source = 'alpha_vantage'
  and symbol is not null
  and source_date is not null
  and open is not null
  and high is not null
  and low is not null
  and close is not null
  and volume is not null
  and open > 0
  and high > 0
  and low > 0
  and close > 0
  and high >= low
  and volume >= 0

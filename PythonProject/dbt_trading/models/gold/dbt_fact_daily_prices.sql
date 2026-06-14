{{ config(alias='dbt_fact_daily_prices') }}

select
    symbol,
    date,
    open,
    high,
    low,
    close,
    volume,
    current_timestamp as loaded_at
from {{ ref('dbt_clean_daily_prices') }}

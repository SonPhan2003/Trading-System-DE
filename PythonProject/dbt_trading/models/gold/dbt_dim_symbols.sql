{{ config(alias='dbt_dim_symbols') }}

select
    symbol,
    cast(null as char(255)) as company_name,
    'Equity' as asset_type,
    cast(null as char(100)) as exchange_name,
    'USD' as currency,
    true as is_active,
    current_timestamp as created_at,
    current_timestamp as updated_at
from {{ ref('dbt_clean_daily_prices') }}
group by symbol

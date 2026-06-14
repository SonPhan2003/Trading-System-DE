select
    symbol,
    date,
    count(*) as row_count
from {{ ref('dbt_fact_daily_prices') }}
group by symbol, date
having count(*) > 1

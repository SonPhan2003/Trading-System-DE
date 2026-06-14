select
    symbol,
    date,
    count(*) as row_count
from {{ ref('dbt_clean_daily_prices') }}
group by symbol, date
having count(*) > 1

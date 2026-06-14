select *
from {{ ref('dbt_clean_daily_prices') }}
where open <= 0
   or high <= 0
   or low <= 0
   or close <= 0
   or high < low
   or volume < 0

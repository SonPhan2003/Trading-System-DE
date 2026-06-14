FROM python:3.9-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir dbt-mysql==1.7.0

WORKDIR /usr/app

ENV DBT_PROFILES_DIR=/usr/app

CMD ["dbt", "--help"]

FROM apache/airflow:2.9.3-python3.11

USER root
RUN apt-get update -qq && \
    apt-get install -y -qq libgomp1 gcc g++ && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/*

USER airflow
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

USER root
RUN chown -R airflow: /opt/airflow
USER airflow

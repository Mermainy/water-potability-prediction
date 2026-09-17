FROM docker:27.5.1-cli AS docker-cli
FROM apache/airflow:3.3.1-python3.11
USER root
COPY --from=docker-cli /usr/local/bin/docker /usr/local/bin/docker
COPY --from=docker-cli /usr/local/libexec/docker/cli-plugins/ /usr/local/libexec/docker/cli-plugins/
USER airflow
RUN pip install --no-cache-dir "apache-airflow[fab,standard]==3.3.1" \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.3.1/constraints-3.11.txt" \
    && pip check
COPY requirements-model.txt /tmp/requirements-model.txt
# Isolate ML libraries from Airflow's preinstalled provider dependencies.
RUN python -m venv /home/airflow/ml-env \
    && /home/airflow/ml-env/bin/pip install --no-cache-dir -r /tmp/requirements-model.txt \
    && /home/airflow/ml-env/bin/pip check
ENV ML_PYTHON=/home/airflow/ml-env/bin/python

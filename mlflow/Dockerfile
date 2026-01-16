FROM python:3.11-slim

# Устанавливаем curl для healthcheck
USER root
RUN apt-get update -qq && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

# Python зависимости
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# MLflow директории
RUN mkdir -p /mlflow/mlruns
EXPOSE 5000

CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:///mlflow.db", \
     "--default-artifact-root", "/mlruns", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
FROM python:3.11-slim

USER root
RUN apt-get update -qq && \
    apt-get install -y -qq libgomp1 gcc g++ && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5001
CMD ["python", "app.py"]

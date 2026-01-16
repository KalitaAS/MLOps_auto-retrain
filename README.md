# MLOps_auto-retrain

## Описание проекта

Проект реализует **MLOps pipeline** с автоматическим **мониторингом Data Drift** и **переподготовкой модели** при обнаружении изменений в данных.

Основные компоненты:

* мониторинг data drift,
* автоматический retraining,
* оркестрация через Airflow,
* логирование и registry моделей в MLflow,
* REST API для предсказаний (Flask),
* запуск через Docker.

---

## Структура проекта

```
MLOps_auto-retrain/
├── dags/
│ ├── pycache/
│ │ └── drift_monitoring.cpython-311....pyc
│ └── drift_monitoring.py
├── data/
│ ├── Iris.csv
│ └── drift.csv
├── flask-api/
│ ├── logs/
│ ├── Dockerfile
│ ├── app.py
│ └── requirements.txt
├── mlflow/
│ ├── Dockerfile
│ ├── requirements.txt
│ └── docker-compose.yml
└── README.md
```

---

## Pipeline

1. **Data Drift Monitoring**

   * Сравнение train и current данных
   * При превышении порога — retraining

2. **Airflow**

   * DAG запускается по расписанию
   * Управляет проверкой drift и обучением

3. **Model Training + MLflow**

   * Логирование метрик и моделей
   * Использование последней production-версии

4. **Flask API**

   * Эндпоинт `POST /predict`
   * Загружает модель из MLflow

---

## Запуск проекта

```bash
git clone https://github.com/KalitaAS/MLOps_auto-retrain.git
cd MLOps_auto-retrain
docker-compose up --build
```

---

## Доступ к сервисам

| Сервис     | URL                                            |
| ---------- | ---------------------------------------------- |
| Airflow UI | [http://localhost:8080](http://localhost:8080) |
| MLflow UI  | [http://localhost:5000](http://localhost:5000) |
| Flask API  | [http://localhost:8000](http://localhost:8000) |

---

## Пример запроса

```bash
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{"features": {...}}'
```

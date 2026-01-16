from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.utils.dates import days_ago
import pandas as pd
import numpy as np
import logging
import os
import mlflow
from pycaret.classification import setup, compare_models, finalize_model, save_model, pull
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("iris_drift_detection")

DATA_PATH = "/opt/airflow/data"

dag = DAG(
    'drift_monitoring',
    default_args={'retries': 2},
    description='Daily Iris data drift check, PyCaret retraining + MLflow registry',
    schedule_interval='@daily',
    start_date=days_ago(1),
    catchup=False,
    tags=['mlops', 'drift', 'iris', 'mlflow']
)

def psi(expected, actual, buckets=10):
    expected = np.array(expected)
    actual = np.array(actual)
    
    breaks = np.linspace(expected.min(), expected.max(), buckets + 1)
    
    exp_hist, _ = np.histogram(expected, bins=breaks)
    act_hist, _ = np.histogram(actual, bins=breaks)
    
    exp_prop = exp_hist / len(expected)
    act_prop = act_hist / len(actual)
    
    psi_val = np.sum((exp_prop - act_prop) * np.log((exp_prop + 1e-15) / (act_prop + 1e-15)))
    return abs(psi_val)

def load_reference_and_current_data(**context):
    try:
        ref_path = os.path.join(DATA_PATH, 'Iris.csv')
        cur_path = os.path.join(DATA_PATH, 'drift.csv')
        
        logger.info(f"Looking for data at: {ref_path}")
        logger.info(f"Looking for data at: {cur_path}")
        
        if not os.path.exists(ref_path):
            raise FileNotFoundError(f"Reference data not found at {ref_path}")
        if not os.path.exists(cur_path):
            raise FileNotFoundError(f"Current data not found at {cur_path}")
        
        reference = pd.read_csv(ref_path)
        current = pd.read_csv(cur_path)
        
        logger.info(f"Reference data shape: {reference.shape}")
        logger.info(f"Current data shape: {current.shape}")
        
        if 'species' not in reference.columns:
            raise ValueError("'species' column not found in reference data")
        if 'species' not in current.columns:
            raise ValueError("'species' column not found in current data")
        
        context['ti'].xcom_push(key='reference_data', value=reference.to_dict('records'))
        context['ti'].xcom_push(key='current_data', value=current.to_dict('records'))
        
        feature_cols = [col for col in reference.columns if col != 'species']
        context['ti'].xcom_push(key='features', value=feature_cols)
        
        logger.info(f"Loaded {len(reference)} reference, {len(current)} current samples")
        logger.info(f"Features: {feature_cols}")
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def check_drift(**context):
    try:
        ref_records = context['ti'].xcom_pull(key='reference_data', task_ids='load_reference_and_current_data')
        cur_records = context['ti'].xcom_pull(key='current_data', task_ids='load_reference_and_current_data')
        features = context['ti'].xcom_pull(key='features', task_ids='load_reference_and_current_data')
        
        if not ref_records or not cur_records:
            raise ValueError("No data loaded from XCom")
        
        reference = pd.DataFrame(ref_records)
        current = pd.DataFrame(cur_records)
        
        max_psi = 0
        psi_results = {}
        
        for feature in features:
            if feature not in reference.columns or feature not in current.columns:
                logger.warning(f"Feature {feature} not found in data")
                continue
                
            psi_val = psi(reference[feature], current[feature])
            psi_results[feature] = psi_val
            max_psi = max(max_psi, psi_val)
            logger.info(f"PSI {feature}: {psi_val:.4f}")
        
        drift_detected = max_psi > 0.2
        
        logger.info(f"Max PSI: {max_psi:.4f}")
        logger.info(f"Drift detected: {drift_detected}")
        
        context['ti'].xcom_push(key='drift_detected', value=drift_detected)
        context['ti'].xcom_push(key='max_psi', value=max_psi)
        context['ti'].xcom_push(key='psi_results', value=psi_results)
        
        return 'retrain_model' if drift_detected else 'log_no_drift_detected'
        
    except Exception as e:
        logger.error(f"Error checking drift: {e}")
        raise

def retrain_and_log_mlflow(**context):
    try:        
        cur_records = context['ti'].xcom_pull(key='current_data', task_ids='load_reference_and_current_data')
        current = pd.DataFrame(cur_records)
        
        logger.info(f"Starting retraining with {len(current)} samples")
        
        with mlflow.start_run(run_name=f"iris_retrained_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            logger.info("Setting up PyCaret...")
            
            try:
                s = setup(
                    data=current, 
                    target='species', 
                    train_size=0.8, 
                    session_id=42,
                    verbose=False,
                    log_experiment=True,
                    experiment_name='iris_drift_detection',
                    log_plots=False
                )
                
                logger.info("Comparing models...")
                best_model = compare_models(
                    n_select=1,
                    fold=3,
                    verbose=False
                )
                
                final_model = finalize_model(best_model)
                
                metrics_df = pull()
                logger.info("\nPyCaret Metrics Summary:")
                for col in metrics_df.columns:
                    value = metrics_df[col].iloc[0]
                    if isinstance(value, (float)):
                        logger.info(f"{col}: {value:.4f}")
                    else:
                        logger.info(f"{col}: {value}")
                
                accuracy = float(metrics_df['Accuracy'].iloc[0])
                recall = float(metrics_df['Recall'].iloc[0])
                
                mlflow.log_params(best_model.get_params())
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("train_samples", len(current))
                
                mlflow.sklearn.log_model(
                    final_model, 
                    artifact_path="iris_model",
                    registered_model_name="iris_model"
                )
                
                model_path = '/opt/airflow/models'
                os.makedirs(model_path, exist_ok=True)
                save_model(final_model, os.path.join(model_path, f"iris_retrained_{datetime.now().strftime('%Y%m%d')}.pkl"))
                
                run_id = mlflow.active_run().info.run_id
                model_uri = f"runs:/{run_id}/iris_model"
                
                context['ti'].xcom_push(key='mlflow_run_id', value=run_id)
                context['ti'].xcom_push(key='model_uri', value=model_uri)
                context['ti'].xcom_push(key='accuracy', value=accuracy)
                
                logger.info(f"MLflow run: {run_id}")
                logger.info(f"Model URI: {model_uri}")
                logger.info(f"Accuracy: {accuracy:.4f}")
                
            except Exception as e:
                logger.error(f"PyCaret error: {e}")
                raise
                
    except Exception as e:
        logger.error(f"Error in retraining: {e}")
        raise

def register_staging_model(**context):
    try:
        run_id = context['ti'].xcom_pull(key='mlflow_run_id', task_ids='retrain_model')
        model_uri = context['ti'].xcom_pull(key='model_uri', task_ids='retrain_model')
        accuracy = context['ti'].xcom_pull(key='accuracy', task_ids='retrain_model')
        
        if not run_id or not accuracy:
            logger.warning("No run_id or accuracy found. Skipping registration.")
            return
        
        logger.info(f"Registering model from run {run_id} with accuracy {accuracy:.4f}")
        
        client = mlflow.MlflowClient()
        
        model_name = "iris_model"
        
        if accuracy > 0.90:
            model_version = mlflow.register_model(model_uri, model_name)
            
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Staging"
            )
            
            client.update_model_version(
                name=model_name,
                version=model_version.version,
                description=f"Iris classifier retrained. Accuracy: {accuracy:.4f}"
            )
            
            context['ti'].xcom_push(key='model_version', value=model_version.version)
            logger.info(f"Model version {model_version.version} registered and moved to STAGING")
        else:
            logger.warning(f"Model accuracy {accuracy:.4f} below threshold (0.85). Skipping registration.")
            
    except Exception as e:
        logger.error(f"Error in registration: {e}")

def log_no_drift_detected(**context):
    psi_val = context['ti'].xcom_pull(key='max_psi', task_ids='check_drift')
    logger.info(f"No drift detected. Max PSI: {psi_val:.4f}")

def promote_to_production(**context):
    version = context['ti'].xcom_pull(key='model_version', task_ids='register_staging')
    if not version:
        logger.info("No model version to promote. Skipping promotion.")
        return

    client = mlflow.MlflowClient()
    model_name = "iris_model"

    try:
        # Поиск всех версии, находящиеся в Production
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        
        # Архивировать каждую из них
        for v in prod_versions:
            logger.info(f"Archiving current Production model version {v.version}")
            client.transition_model_version_stage(
                name=model_name,
                version=v.version,
                stage="Archived"
            )

        # Продвинуть новую версию в Production
        logger.info(f"Promoting new model version {version} to Production")
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )
        logger.info(f"Model v{version} successfully promoted to PRODUCTION")

    except Exception as e:
        logger.error(f"Error during promotion to Production: {e}")
        raise

load_task = PythonOperator(task_id='load_reference_and_current_data', python_callable=load_reference_and_current_data, dag=dag)
check_task = BranchPythonOperator(task_id='check_drift', python_callable=check_drift, dag=dag)
retrain_task = PythonOperator(task_id='retrain_model', python_callable=retrain_and_log_mlflow, dag=dag)
register_task = PythonOperator(task_id='register_staging', python_callable=register_staging_model, dag=dag)
no_drift_task = PythonOperator(task_id='no_drift_detected', python_callable=log_no_drift_detected, dag=dag)
promote_task = PythonOperator(task_id='promote_production', python_callable=promote_to_production, dag=dag)


load_task >> check_task
check_task >> [retrain_task, no_drift_task]
retrain_task >> register_task
register_task >> promote_task
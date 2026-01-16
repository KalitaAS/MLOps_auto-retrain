from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
import json

app = Flask(__name__)

api = Api(app, 
          version='1.0', 
          title='Iris A/B Testing API',
          description='API для A/B тестирования моделей Iris из MLFlow',
          doc='/docs/')

mlflow.set_tracking_uri("http://mlflow:5000")
MODEL_NAME = "iris_model"
B_TRAFFIC_RATIO = float(os.getenv('B_TRAFFIC_RATIO', '0.3'))
FEATURE_NAMES = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(f'./logs/flask_api.log')
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

feature_model = api.model('Features', {
    'features': fields.List(fields.Float, 
                          required=True, 
                          description='Список из 4 признаков Iris',
                          example=[5.1, 3.5, 1.4, 0.2])
})

prediction_model = api.model('Prediction', {
    'prediction': fields.String(description='Предсказанный класс', example='Iris-setosa'),
    'model_stage': fields.String(description='Использованная модель', example='Production'),
    'traffic_split': fields.Float(description='Текущий сплит трафика', example=0.3)
})

traffic_model = api.model('Traffic', {
    'b_ratio': fields.Float(required=True, 
                           description='Новое значение для трафика модели B (0-1)',
                           min=0, max=1, example=0.3)
})

def load_model(stage):
    try:
        model_uri = f"models:/{MODEL_NAME}/{stage}"
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Загружена модель {stage}")
        return model
    except Exception as e:
        logger.warning(f"Не удалось загрузить модель {stage}: {e}")
        
        if stage.lower() == 'staging':
            try:
                model_uri = f"models:/{MODEL_NAME}/Production"
                model = mlflow.sklearn.load_model(model_uri)
                logger.info("Используется Production вместо Staging")
                return model
            except Exception as e2:
                logger.error(f"Не удалось загрузить Production модель: {e2}")
        
        return None

def prepare_features(features_list):
    try:
        features_array = np.array(features_list).reshape(1, -1)
        features_df = pd.DataFrame(features_array, columns=FEATURE_NAMES)
        
        logger.info(f"Подготовлены признаки: {features_df.to_dict('records')[0]}")
        return features_df
        
    except Exception as e:
        logger.error(f"Ошибка подготовки признаков: {e}")
        raise

@api.route('/health')
class Health(Resource):
    def get(self):
        return {"status": "healthy"}

@api.route('/predict')
class Predict(Resource):
    @api.expect(feature_model)
    @api.marshal_with(prediction_model)
    def post(self):
        try:
            if not request.is_json:
                return {"error": "Content-Type must be application/json"}, 400
                
            data = request.get_json(force=True)
            features = data['features']
            if not isinstance(features, list) or len(features) != 4:
                return {"error": f"Expected list of 4 features, got {len(features) if isinstance(features, list) else 'N/A'}"}, 400
            
            logger.info(f"Predict request: {features}")
            
            use_staging = np.random.random() < B_TRAFFIC_RATIO
            stage = 'Staging' if use_staging else 'Production'
            
            model = load_model(stage)
            if model is None:
                return {"error": f"Model {stage} not available"}, 503
            
            features_df = prepare_features(features)
            prediction = model.predict(features_df)[0]
            
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'model_stage': stage,
                'features': features,
                'prediction': prediction
            }
            logger.info(json.dumps(log_entry))
            
            return {
                'prediction': prediction,
                'model_stage': stage,
                'traffic_split': B_TRAFFIC_RATIO
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"error": str(e)}, 500

@api.route('/traffic')
class Traffic(Resource):
    @api.expect(traffic_model)
    def post(self):
        global B_TRAFFIC_RATIO
        
        try:
            new_ratio = request.json.get('b_ratio', 0.3)
            B_TRAFFIC_RATIO = max(0, min(1, new_ratio))
            os.environ['B_TRAFFIC_RATIO'] = str(B_TRAFFIC_RATIO)
            
            logger.info(f"Traffic split updated to B: {B_TRAFFIC_RATIO}")
            
            return {
                'status': 'updated', 
                'b_ratio': B_TRAFFIC_RATIO,
                'message': f'Traffic split updated. Model B (Staging) will receive {B_TRAFFIC_RATIO*100:.1f}% of traffic'
            }
        except Exception as e:
            logger.error(f"Traffic update error: {e}")
            return {"error": str(e)}, 500

if __name__ == '__main__':    
    app.run(host='0.0.0.0', port=5001, debug=False)
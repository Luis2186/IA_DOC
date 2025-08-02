---
layout: ../../layouts/DocLayout.astro
title: "Integración de Machine Learning en Software"
description: "Cómo integrar modelos de ML en aplicaciones y sistemas de software"
currentPath: "/machine-learning/integracion-en-software"
---

# Integración de Machine Learning en Software

La integración exitosa de modelos de Machine Learning en aplicaciones de software requiere consideraciones especiales de arquitectura, rendimiento, escalabilidad y mantenimiento que van más allá del desarrollo tradicional de software.

## Arquitecturas de Integración

### 1. Modelo como Servicio (Model as a Service)

#### API REST
```python
# Ejemplo con FastAPI
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
modelo = joblib.load('modelo_entrenado.pkl')

class PredictionRequest(BaseModel):
    features: list[float]

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    # Preparar datos
    input_data = np.array([request.features])
    
    # Realizar predicción
    prediction = modelo.predict(input_data)[0]
    confidence = modelo.predict_proba(input_data).max()
    
    return PredictionResponse(
        prediction=prediction,
        confidence=confidence
    )
```

#### Ventajas
- Separación clara de responsabilidades
- Escalabilidad independiente
- Reutilización entre aplicaciones
- Facilita A/B testing

#### Desventajas
- Latencia de red
- Punto único de falla
- Complejidad de infraestructura

### 2. Modelo Embebido

```python
# Integración directa en la aplicación
import joblib

class ProductRecommendationService:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load('scaler.pkl')
    
    def get_recommendations(self, user_features):
        # Preprocessing
        scaled_features = self.scaler.transform([user_features])
        
        # Predicción
        recommendations = self.model.predict(scaled_features)
        
        return recommendations.tolist()

# En el servicio principal
recommendation_service = ProductRecommendationService('model.pkl')
```

#### Ventajas
- Menor latencia
- Sin dependencias de red
- Simplifica deployment

#### Desventajas
- Acoplamiento fuerte
- Dificulta actualizaciones
- Duplicación de código

### 3. Batch Processing

```python
# Procesamiento por lotes usando Apache Airflow
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

def run_batch_predictions():
    # Cargar datos nuevos
    new_data = load_new_data()
    
    # Cargar modelo
    model = load_model('latest_model.pkl')
    
    # Generar predicciones
    predictions = model.predict(new_data)
    
    # Guardar resultados
    save_predictions(predictions)

dag = DAG(
    'batch_ml_predictions',
    default_args={
        'owner': 'ml-team',
        'retries': 1,
        'retry_delay': timedelta(minutes=5)
    },
    schedule_interval='@daily'
)

prediction_task = PythonOperator(
    task_id='generate_predictions',
    python_callable=run_batch_predictions,
    dag=dag
)
```

#### Ventajas
- Procesamiento eficiente de grandes volúmenes
- Menos recursos en tiempo real
- Fácil paralelización

#### Desventajas
- No apto para decisiones en tiempo real
- Resultados pueden quedar obsoletos
- Complejidad en orquestación

## Patrones de Diseño para ML

### 1. Pipeline Pattern

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier

# Pipeline completo
ml_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(k=10)),
    ('classifier', RandomForestClassifier())
])

# Entrenamiento
ml_pipeline.fit(X_train, y_train)

# Predicción (incluye todo el preprocessing)
predictions = ml_pipeline.predict(X_new)
```

### 2. Model Registry Pattern

```python
import mlflow
import mlflow.sklearn

class ModelRegistry:
    def __init__(self):
        self.models = {}
    
    def register_model(self, name, model, version):
        """Registrar modelo con versionado"""
        model_uri = f"models:/{name}/{version}"
        mlflow.sklearn.log_model(model, name)
        self.models[name] = {
            'model': model,
            'version': version,
            'uri': model_uri
        }
    
    def get_model(self, name, version='latest'):
        """Obtener modelo específico"""
        if version == 'latest':
            return mlflow.sklearn.load_model(f"models:/{name}/latest")
        else:
            return mlflow.sklearn.load_model(f"models:/{name}/{version}")
    
    def promote_model(self, name, version, stage):
        """Promover modelo a producción"""
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=name,
            version=version,
            stage=stage
        )
```

### 3. Feature Store Pattern

```python
class FeatureStore:
    def __init__(self, connection_string):
        self.db = connect(connection_string)
    
    def get_features(self, entity_id, feature_names, timestamp=None):
        """Obtener características para una entidad"""
        if timestamp is None:
            timestamp = datetime.now()
        
        query = f"""
        SELECT {', '.join(feature_names)}
        FROM features
        WHERE entity_id = %s AND timestamp <= %s
        ORDER BY timestamp DESC
        LIMIT 1
        """
        
        return self.db.execute(query, (entity_id, timestamp))
    
    def store_features(self, entity_id, features, timestamp=None):
        """Almacenar características calculadas"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Lógica para guardar características
        pass

# Uso en predicción
feature_store = FeatureStore(connection_string)
features = feature_store.get_features(
    entity_id=user_id,
    feature_names=['age', 'income', 'purchase_history']
)
```

## Consideraciones de Rendimiento

### 1. Optimización de Modelos

#### Cuantización
```python
import tensorflow as tf

# Cuantización post-entrenamiento
converter = tf.lite.TFLiteConverter.from_saved_model('modelo_path')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_model = converter.convert()

# Guardar modelo cuantizado
with open('modelo_cuantizado.tflite', 'wb') as f:
    f.write(quantized_model)
```

#### Pruning (Poda)
```python
import tensorflow_model_optimization as tfmot

# Poda de pesos durante entrenamiento
pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
    initial_sparsity=0.0,
    final_sparsity=0.5,
    begin_step=1000,
    end_step=5000
)

pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
    model,
    pruning_schedule=pruning_schedule
)
```

### 2. Caching de Predicciones

```python
import redis
import json
from hashlib import md5

class PredictionCache:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port)
        self.ttl = 3600  # 1 hora
    
    def get_cache_key(self, features):
        """Generar clave única para características"""
        features_str = json.dumps(features, sort_keys=True)
        return md5(features_str.encode()).hexdigest()
    
    def get_prediction(self, features):
        """Obtener predicción del cache"""
        cache_key = self.get_cache_key(features)
        cached_result = self.redis_client.get(cache_key)
        
        if cached_result:
            return json.loads(cached_result)
        return None
    
    def store_prediction(self, features, prediction):
        """Almacenar predicción en cache"""
        cache_key = self.get_cache_key(features)
        self.redis_client.setex(
            cache_key,
            self.ttl,
            json.dumps(prediction)
        )

# Uso con cache
cache = PredictionCache()

def predict_with_cache(features):
    # Intentar obtener del cache
    cached_prediction = cache.get_prediction(features)
    if cached_prediction:
        return cached_prediction
    
    # Si no está en cache, calcular
    prediction = model.predict([features])
    
    # Guardar en cache
    cache.store_prediction(features, prediction.tolist())
    
    return prediction.tolist()
```

### 3. Optimización de Infraestructura

#### Load Balancing
```yaml
# docker-compose.yml para múltiples instancias
version: '3.8'
services:
  ml-api-1:
    image: ml-api:latest
    ports:
      - "8001:8000"
  
  ml-api-2:
    image: ml-api:latest
    ports:
      - "8002:8000"
  
  nginx:
    image: nginx:latest
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - ml-api-1
      - ml-api-2
```

```nginx
# nginx.conf
upstream ml_backend {
    server ml-api-1:8000;
    server ml-api-2:8000;
}

server {
    listen 80;
    
    location /predict {
        proxy_pass http://ml_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Monitoreo y Observabilidad

### 1. Métricas de Modelo

```python
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge

# Métricas Prometheus
PREDICTION_COUNTER = Counter('ml_predictions_total', 'Total predictions made')
PREDICTION_LATENCY = Histogram('ml_prediction_duration_seconds', 'Prediction latency')
MODEL_ACCURACY = Gauge('ml_model_accuracy', 'Current model accuracy')

def monitored_predict(features):
    with PREDICTION_LATENCY.time():
        prediction = model.predict([features])
        PREDICTION_COUNTER.inc()
        return prediction
```

### 2. Data Drift Detection

```python
from evidently.model_monitoring import ModelMonitoring
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab

def monitor_data_drift(reference_data, current_data):
    """Detectar deriva en los datos"""
    
    monitoring = ModelMonitoring(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping
    )
    
    dashboard = Dashboard(tabs=[DataDriftTab()])
    dashboard.calculate(monitoring)
    
    return dashboard.show()
```

### 3. Logging Estructurado

```python
import logging
import json
from datetime import datetime

class MLLogger:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def log_prediction(self, model_version, features, prediction, confidence):
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'prediction',
            'model_version': model_version,
            'features': features,
            'prediction': prediction,
            'confidence': confidence
        }
        
        self.logger.info(json.dumps(log_entry))
    
    def log_model_update(self, old_version, new_version, metrics):
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'model_update',
            'old_version': old_version,
            'new_version': new_version,
            'metrics': metrics
        }
        
        self.logger.info(json.dumps(log_entry))
```

## Gestión de Versiones

### 1. Versionado de Modelos

```python
class ModelVersionManager:
    def __init__(self, storage_path):
        self.storage_path = storage_path
        self.current_version = self.get_latest_version()
    
    def save_model(self, model, version, metadata):
        """Guardar modelo con versión específica"""
        model_path = f"{self.storage_path}/model_v{version}.pkl"
        metadata_path = f"{self.storage_path}/metadata_v{version}.json"
        
        # Guardar modelo
        joblib.dump(model, model_path)
        
        # Guardar metadatos
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
    
    def load_model(self, version=None):
        """Cargar modelo específico o último"""
        if version is None:
            version = self.current_version
        
        model_path = f"{self.storage_path}/model_v{version}.pkl"
        return joblib.load(model_path)
    
    def rollback(self, target_version):
        """Rollback a versión anterior"""
        self.current_version = target_version
        return self.load_model(target_version)
```

### 2. Blue-Green Deployment

```python
class BlueGreenDeployment:
    def __init__(self):
        self.blue_model = None
        self.green_model = None
        self.current_color = 'blue'
    
    def deploy_new_version(self, new_model):
        """Desplegar nueva versión sin downtime"""
        if self.current_color == 'blue':
            # Desplegar en green
            self.green_model = new_model
            
            # Verificar que funciona
            if self.health_check(self.green_model):
                self.current_color = 'green'
                return True
        else:
            # Desplegar en blue
            self.blue_model = new_model
            
            if self.health_check(self.blue_model):
                self.current_color = 'blue'
                return True
        
        return False
    
    def get_current_model(self):
        """Obtener modelo activo"""
        if self.current_color == 'blue':
            return self.blue_model
        else:
            return self.green_model
    
    def health_check(self, model):
        """Verificar salud del modelo"""
        try:
            # Predicción de prueba
            test_prediction = model.predict([[1, 2, 3, 4]])
            return True
        except Exception:
            return False
```

## Seguridad y Compliance

### 1. Autenticación y Autorización

```python
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

app = FastAPI()
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(
            credentials.credentials,
            "secret_key",
            algorithms=["HS256"]
        )
        return payload
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/predict")
async def secure_predict(
    request: PredictionRequest,
    user = Depends(verify_token)
):
    # Verificar permisos
    if not user.get('can_predict'):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    # Realizar predicción
    prediction = model.predict([request.features])
    return {"prediction": prediction[0]}
```

### 2. Privacidad de Datos

```python
import hashlib

class PrivacyPreservingPredictor:
    def __init__(self, model):
        self.model = model
    
    def hash_sensitive_features(self, features):
        """Hash de características sensibles"""
        # Identificar características sensibles
        sensitive_indices = [0, 2, 5]  # ej: nombre, SSN, dirección
        
        hashed_features = features.copy()
        for idx in sensitive_indices:
            if idx < len(features):
                # Hash irreversible
                hashed_features[idx] = hashlib.sha256(
                    str(features[idx]).encode()
                ).hexdigest()[:8]
        
        return hashed_features
    
    def predict_with_privacy(self, features):
        """Predicción preservando privacidad"""
        hashed_features = self.hash_sensitive_features(features)
        return self.model.predict([hashed_features])
```

## Mejores Prácticas

### 1. Testing de Modelos

```python
import pytest
import numpy as np

class TestMLModel:
    def setup_method(self):
        self.model = load_model('test_model.pkl')
        self.test_data = load_test_data()
    
    def test_prediction_shape(self):
        """Verificar forma de salida"""
        prediction = self.model.predict(self.test_data[:1])
        assert prediction.shape == (1,)
    
    def test_prediction_range(self):
        """Verificar rango de predicciones"""
        predictions = self.model.predict(self.test_data)
        assert np.all(predictions >= 0)
        assert np.all(predictions <= 1)
    
    def test_model_consistency(self):
        """Verificar consistencia"""
        pred1 = self.model.predict(self.test_data[:1])
        pred2 = self.model.predict(self.test_data[:1])
        np.testing.assert_array_equal(pred1, pred2)
    
    def test_model_bias(self):
        """Verificar sesgos"""
        # Datos por grupo demográfico
        group_a = self.test_data[self.test_data['group'] == 'A']
        group_b = self.test_data[self.test_data['group'] == 'B']
        
        pred_a = self.model.predict(group_a)
        pred_b = self.model.predict(group_b)
        
        # Verificar diferencia no significativa
        assert abs(pred_a.mean() - pred_b.mean()) < 0.1
```

### 2. Configuración por Ambientes

```python
# config.py
import os
from dataclasses import dataclass

@dataclass
class MLConfig:
    model_path: str
    batch_size: int
    max_requests_per_minute: int
    cache_ttl: int
    log_level: str

def get_config():
    env = os.getenv('ENVIRONMENT', 'development')
    
    if env == 'production':
        return MLConfig(
            model_path='s3://models/production/model.pkl',
            batch_size=1000,
            max_requests_per_minute=10000,
            cache_ttl=3600,
            log_level='INFO'
        )
    elif env == 'staging':
        return MLConfig(
            model_path='s3://models/staging/model.pkl',
            batch_size=100,
            max_requests_per_minute=1000,
            cache_ttl=1800,
            log_level='DEBUG'
        )
    else:  # development
        return MLConfig(
            model_path='./models/dev_model.pkl',
            batch_size=10,
            max_requests_per_minute=100,
            cache_ttl=300,
            log_level='DEBUG'
        )
```

La integración exitosa de ML en software requiere un enfoque holístico que considere no solo la precisión del modelo, sino también aspectos operacionales como rendimiento, escalabilidad, monitoreo y mantenimiento.

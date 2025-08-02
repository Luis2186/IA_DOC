---
layout: ../../layouts/DocLayout.astro
title: "Ejemplos Prácticos de Machine Learning"
description: "Casos de uso reales y implementaciones paso a paso"
currentPath: "/machine-learning/ejemplos"
---

# Ejemplos Prácticos de Machine Learning

Esta sección presenta implementaciones completas de casos de uso reales de Machine Learning, desde la preparación de datos hasta el despliegue del modelo.

## Ejemplo 1: Sistema de Recomendación de Productos

### Contexto
Crear un sistema que recomiende productos a usuarios basándose en su historial de compras y comportamiento de navegación.

### Datos
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Cargar datos de ejemplo
users = pd.read_csv('users.csv')  # user_id, age, gender, location
products = pd.read_csv('products.csv')  # product_id, category, price, brand
interactions = pd.read_csv('interactions.csv')  # user_id, product_id, rating, timestamp

print("Datos cargados:")
print(f"Usuarios: {len(users)}")
print(f"Productos: {len(products)}")
print(f"Interacciones: {len(interactions)}")
```

### Preparación de Datos
```python
# Función para crear características de usuario
def create_user_features(user_id, interactions, products):
    user_interactions = interactions[interactions['user_id'] == user_id]
    
    if len(user_interactions) == 0:
        return {
            'avg_rating': 0,
            'num_interactions': 0,
            'favorite_category': 'unknown',
            'avg_price_range': 0
        }
    
    # Unir con información de productos
    user_products = user_interactions.merge(products, on='product_id')
    
    features = {
        'avg_rating': user_products['rating'].mean(),
        'num_interactions': len(user_interactions),
        'favorite_category': user_products['category'].mode().iloc[0] if len(user_products) > 0 else 'unknown',
        'avg_price_range': user_products['price'].mean()
    }
    
    return features

# Crear dataset de entrenamiento
def prepare_training_data():
    training_data = []
    
    for _, interaction in interactions.iterrows():
        user_id = interaction['user_id']
        product_id = interaction['product_id']
        rating = interaction['rating']
        
        # Características del usuario
        user_features = create_user_features(user_id, interactions, products)
        
        # Características del producto
        product_info = products[products['product_id'] == product_id].iloc[0]
        
        # Combinar características
        features = {
            **user_features,
            'product_price': product_info['price'],
            'product_category': product_info['category'],
            'target_rating': rating
        }
        
        training_data.append(features)
    
    return pd.DataFrame(training_data)

# Preparar datos
df_training = prepare_training_data()

# Encoding de variables categóricas
df_encoded = pd.get_dummies(df_training, columns=['favorite_category', 'product_category'])

print(f"Dataset preparado: {df_encoded.shape}")
print(df_encoded.head())
```

### Entrenamiento del Modelo
```python
# Separar características y objetivo
X = df_encoded.drop(['target_rating'], axis=1)
y = df_encoded['target_rating']

# División train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Entrenar modelo
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluación
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")

# Importancia de características
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 características más importantes:")
print(feature_importance.head(10))
```

### Sistema de Recomendación
```python
class ProductRecommendationSystem:
    def __init__(self, model, products_df, interactions_df):
        self.model = model
        self.products = products_df
        self.interactions = interactions_df
        self.feature_columns = model.feature_names_in_
    
    def get_recommendations(self, user_id, n_recommendations=5):
        # Obtener características del usuario
        user_features = create_user_features(user_id, self.interactions, self.products)
        
        # Productos que el usuario no ha interactuado
        user_products = self.interactions[
            self.interactions['user_id'] == user_id
        ]['product_id'].unique()
        
        unseen_products = self.products[
            ~self.products['product_id'].isin(user_products)
        ]
        
        recommendations = []
        
        for _, product in unseen_products.iterrows():
            # Crear características para predicción
            features = {
                **user_features,
                'product_price': product['price'],
                'product_category': product['category']
            }
            
            # Convertir a formato del modelo
            feature_vector = self.prepare_features(features)
            
            # Predecir rating
            predicted_rating = self.model.predict([feature_vector])[0]
            
            recommendations.append({
                'product_id': product['product_id'],
                'predicted_rating': predicted_rating,
                'product_name': product.get('name', f"Product {product['product_id']}"),
                'price': product['price'],
                'category': product['category']
            })
        
        # Ordenar por rating predicho
        recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
        
        return recommendations[:n_recommendations]
    
    def prepare_features(self, features_dict):
        # Crear vector de características compatible con el modelo
        feature_vector = np.zeros(len(self.feature_columns))
        
        for i, col in enumerate(self.feature_columns):
            if col in features_dict:
                feature_vector[i] = features_dict[col]
            elif col.startswith('favorite_category_') or col.startswith('product_category_'):
                # Manejo de variables dummy
                category = col.split('_', 1)[1]
                if col.startswith('favorite_category_'):
                    if features_dict.get('favorite_category') == category:
                        feature_vector[i] = 1
                else:  # product_category_
                    if features_dict.get('product_category') == category:
                        feature_vector[i] = 1
        
        return feature_vector

# Crear sistema de recomendación
recommendation_system = ProductRecommendationSystem(model, products, interactions)

# Ejemplo de uso
user_id = 123
recommendations = recommendation_system.get_recommendations(user_id, n_recommendations=5)

print(f"\nRecomendaciones para usuario {user_id}:")
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec['product_name']} - Rating predicho: {rec['predicted_rating']:.2f}")
```

## Ejemplo 2: Detección de Fraude en Transacciones

### Contexto
Desarrollar un sistema que detecte transacciones fraudulentas en tiempo real basándose en patrones históricos.

### Preparación de Datos
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Cargar datos de transacciones
transactions = pd.read_csv('transactions.csv')
# Columnas: transaction_id, user_id, amount, merchant, timestamp, is_fraud

print(f"Total transacciones: {len(transactions)}")
print(f"Transacciones fraudulentas: {transactions['is_fraud'].sum()}")
print(f"Porcentaje de fraude: {transactions['is_fraud'].mean()*100:.2f}%")

# Ingeniería de características
def create_fraud_features(df):
    df = df.copy()
    
    # Convertir timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Características por usuario
    user_stats = df.groupby('user_id').agg({
        'amount': ['mean', 'std', 'max'],
        'transaction_id': 'count'
    }).round(2)
    
    user_stats.columns = ['user_avg_amount', 'user_std_amount', 'user_max_amount', 'user_transaction_count']
    user_stats = user_stats.reset_index()
    
    df = df.merge(user_stats, on='user_id', how='left')
    
    # Desviación de la transacción respecto al comportamiento del usuario
    df['amount_deviation'] = abs(df['amount'] - df['user_avg_amount']) / (df['user_std_amount'] + 1e-8)
    
    # Características temporales
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
    
    # Características por comercio
    merchant_stats = df.groupby('merchant').agg({
        'is_fraud': 'mean',
        'amount': 'mean'
    }).round(4)
    
    merchant_stats.columns = ['merchant_fraud_rate', 'merchant_avg_amount']
    merchant_stats = merchant_stats.reset_index()
    
    df = df.merge(merchant_stats, on='merchant', how='left')
    
    return df

# Aplicar ingeniería de características
df_features = create_fraud_features(transactions)

# Seleccionar características para el modelo
feature_columns = [
    'amount', 'hour', 'day_of_week', 'is_weekend', 'is_night',
    'user_avg_amount', 'user_std_amount', 'user_max_amount', 
    'user_transaction_count', 'amount_deviation',
    'merchant_fraud_rate', 'merchant_avg_amount'
]

X = df_features[feature_columns].fillna(0)
y = df_features['is_fraud']

print(f"Características creadas: {X.shape[1]}")
```

### Modelo de Detección
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve

# División de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Normalización
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Balancear clases con SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print(f"Distribución original: {np.bincount(y_train)}")
print(f"Distribución balanceada: {np.bincount(y_train_balanced)}")

# Entrenar modelos
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'Isolation Forest': IsolationForest(contamination=0.1, random_state=42)
}

results = {}

for name, model in models.items():
    if name == 'Isolation Forest':
        # Isolation Forest es no supervisado
        model.fit(X_train_scaled)
        y_pred = model.predict(X_test_scaled)
        y_pred = (y_pred == -1).astype(int)  # -1 = anomalía, 1 = normal
    else:
        model.fit(X_train_balanced, y_train_balanced)
        y_pred = model.predict(X_test_scaled)
    
    # Métricas
    auc = roc_auc_score(y_test, y_pred) if name != 'Isolation Forest' else roc_auc_score(y_test, y_pred)
    
    results[name] = {
        'model': model,
        'predictions': y_pred,
        'auc': auc,
        'classification_report': classification_report(y_test, y_pred)
    }
    
    print(f"\n{name}:")
    print(f"AUC: {auc:.4f}")
    print(results[name]['classification_report'])

# Seleccionar mejor modelo
best_model_name = max(results.keys(), key=lambda k: results[k]['auc'])
best_model = results[best_model_name]['model']

print(f"\nMejor modelo: {best_model_name}")
```

### Sistema de Detección en Tiempo Real
```python
import joblib
from datetime import datetime, timedelta

class FraudDetectionSystem:
    def __init__(self, model, scaler, feature_columns, threshold=0.5):
        self.model = model
        self.scaler = scaler
        self.feature_columns = feature_columns
        self.threshold = threshold
        self.user_history = {}  # Cache de historial de usuarios
    
    def update_user_history(self, user_id, transaction):
        """Actualizar historial del usuario"""
        if user_id not in self.user_history:
            self.user_history[user_id] = []
        
        self.user_history[user_id].append(transaction)
        
        # Mantener solo transacciones de los últimos 30 días
        cutoff_date = datetime.now() - timedelta(days=30)
        self.user_history[user_id] = [
            t for t in self.user_history[user_id] 
            if t['timestamp'] > cutoff_date
        ]
    
    def calculate_user_features(self, user_id):
        """Calcular características del usuario basadas en historial"""
        if user_id not in self.user_history or not self.user_history[user_id]:
            return {
                'user_avg_amount': 0,
                'user_std_amount': 0,
                'user_max_amount': 0,
                'user_transaction_count': 0
            }
        
        amounts = [t['amount'] for t in self.user_history[user_id]]
        
        return {
            'user_avg_amount': np.mean(amounts),
            'user_std_amount': np.std(amounts),
            'user_max_amount': np.max(amounts),
            'user_transaction_count': len(amounts)
        }
    
    def detect_fraud(self, transaction):
        """Detectar si una transacción es fraudulenta"""
        # Extraer características de la transacción
        timestamp = datetime.strptime(transaction['timestamp'], '%Y-%m-%d %H:%M:%S')
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Características temporales
        features = {
            'amount': transaction['amount'],
            'hour': hour,
            'day_of_week': day_of_week,
            'is_weekend': 1 if day_of_week in [5, 6] else 0,
            'is_night': 1 if hour >= 22 or hour <= 6 else 0,
        }
        
        # Características del usuario
        user_features = self.calculate_user_features(transaction['user_id'])
        features.update(user_features)
        
        # Desviación del monto
        user_avg = user_features['user_avg_amount']
        user_std = user_features['user_std_amount']
        features['amount_deviation'] = abs(transaction['amount'] - user_avg) / (user_std + 1e-8)
        
        # Características del comercio (simulated)
        features['merchant_fraud_rate'] = 0.01  # Default
        features['merchant_avg_amount'] = 100.0  # Default
        
        # Crear vector de características
        feature_vector = np.array([features[col] for col in self.feature_columns]).reshape(1, -1)
        
        # Normalizar
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        # Predicción
        fraud_probability = self.model.predict_proba(feature_vector_scaled)[0][1]
        is_fraud = fraud_probability > self.threshold
        
        # Actualizar historial si no es fraude
        if not is_fraud:
            self.update_user_history(transaction['user_id'], {
                'amount': transaction['amount'],
                'timestamp': timestamp
            })
        
        return {
            'is_fraud': is_fraud,
            'fraud_probability': fraud_probability,
            'risk_factors': self.analyze_risk_factors(features, fraud_probability)
        }
    
    def analyze_risk_factors(self, features, fraud_probability):
        """Analizar factores de riesgo"""
        risk_factors = []
        
        if features['is_night']:
            risk_factors.append("Transacción nocturna")
        
        if features['is_weekend']:
            risk_factors.append("Transacción en fin de semana")
        
        if features['amount_deviation'] > 3:
            risk_factors.append("Monto muy diferente al patrón habitual")
        
        if features['amount'] > 1000:
            risk_factors.append("Monto elevado")
        
        return risk_factors

# Crear sistema de detección
fraud_detector = FraudDetectionSystem(
    model=best_model,
    scaler=scaler,
    feature_columns=feature_columns,
    threshold=0.7
)

# Ejemplo de uso
new_transaction = {
    'user_id': 'user_123',
    'amount': 2500.0,
    'merchant': 'Online Store',
    'timestamp': '2024-08-02 23:30:00'
}

result = fraud_detector.detect_fraud(new_transaction)

print(f"Transacción fraudulenta: {result['is_fraud']}")
print(f"Probabilidad de fraude: {result['fraud_probability']:.4f}")
print(f"Factores de riesgo: {result['risk_factors']}")
```

## Ejemplo 3: Análisis de Sentimientos en Reseñas

### Contexto
Analizar el sentimiento de reseñas de productos para obtener insights sobre la satisfacción del cliente.

### Preparación de Datos
```python
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# Descargar recursos de NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Cargar datos
reviews = pd.read_csv('product_reviews.csv')
# Columnas: review_id, product_id, user_id, rating, review_text

print(f"Total reseñas: {len(reviews)}")
print(f"Distribución de ratings:")
print(reviews['rating'].value_counts().sort_index())

# Preprocesamiento de texto
def preprocess_text(text):
    """Limpiar y preparar texto"""
    if pd.isna(text):
        return ""
    
    # Convertir a minúsculas
    text = text.lower()
    
    # Remover caracteres especiales
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenizar
    tokens = word_tokenize(text)
    
    # Remover stopwords
    stop_words = set(stopwords.words('spanish'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    
    return ' '.join(tokens)

# Aplicar preprocesamiento
reviews['cleaned_text'] = reviews['review_text'].apply(preprocess_text)

# Crear etiquetas de sentimiento
def rating_to_sentiment(rating):
    if rating <= 2:
        return 'negativo'
    elif rating == 3:
        return 'neutral'
    else:
        return 'positivo'

reviews['sentiment'] = reviews['rating'].apply(rating_to_sentiment)

print(f"\nDistribución de sentimientos:")
print(reviews['sentiment'].value_counts())
```

### Entrenamiento del Modelo
```python
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

# Preparar datos
X = reviews['cleaned_text']
y = reviews['sentiment']

# División de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Crear pipeline con vectorización TF-IDF
pipeline_nb = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ('classifier', MultinomialNB())
])

pipeline_svm = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ('classifier', LinearSVC(random_state=42))
])

# Entrenar y evaluar modelos
models = {
    'Naive Bayes': pipeline_nb,
    'SVM': pipeline_svm
}

for name, model in models.items():
    # Validación cruzada
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
    
    # Entrenamiento final
    model.fit(X_train, y_train)
    
    # Evaluación en test
    y_pred = model.predict(X_test)
    
    print(f"\n{name}:")
    print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(classification_report(y_test, y_pred))

# Seleccionar mejor modelo
best_model = pipeline_svm  # Basado en resultados
```

### Sistema de Análisis de Sentimientos
```python
class SentimentAnalysisSystem:
    def __init__(self, model):
        self.model = model
        self.sentiment_mapping = {
            'positivo': 1,
            'neutral': 0,
            'negativo': -1
        }
    
    def analyze_sentiment(self, text):
        """Analizar sentimiento de un texto"""
        cleaned_text = preprocess_text(text)
        
        if not cleaned_text:
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'score': 0
            }
        
        # Predicción
        prediction = self.model.predict([cleaned_text])[0]
        
        # Probabilidades si están disponibles
        try:
            probabilities = self.model.predict_proba([cleaned_text])[0]
            confidence = max(probabilities)
        except AttributeError:
            confidence = 1.0  # SVM no tiene predict_proba por defecto
        
        return {
            'sentiment': prediction,
            'confidence': confidence,
            'score': self.sentiment_mapping[prediction]
        }
    
    def analyze_product_sentiment(self, product_id, reviews_df):
        """Analizar sentimiento de todas las reseñas de un producto"""
        product_reviews = reviews_df[reviews_df['product_id'] == product_id]
        
        if len(product_reviews) == 0:
            return {
                'overall_sentiment': 'neutral',
                'sentiment_distribution': {},
                'total_reviews': 0,
                'average_score': 0
            }
        
        sentiments = []
        scores = []
        
        for text in product_reviews['review_text']:
            result = self.analyze_sentiment(text)
            sentiments.append(result['sentiment'])
            scores.append(result['score'])
        
        # Distribución de sentimientos
        sentiment_counts = pd.Series(sentiments).value_counts()
        sentiment_distribution = sentiment_counts.to_dict()
        
        # Sentimiento general
        avg_score = np.mean(scores)
        if avg_score > 0.2:
            overall_sentiment = 'positivo'
        elif avg_score < -0.2:
            overall_sentiment = 'negativo'
        else:
            overall_sentiment = 'neutral'
        
        return {
            'overall_sentiment': overall_sentiment,
            'sentiment_distribution': sentiment_distribution,
            'total_reviews': len(product_reviews),
            'average_score': avg_score,
            'positive_percentage': sentiment_distribution.get('positivo', 0) / len(sentiments) * 100,
            'negative_percentage': sentiment_distribution.get('negativo', 0) / len(sentiments) * 100
        }
    
    def get_sentiment_insights(self, reviews_df):
        """Obtener insights generales de sentimiento"""
        insights = {}
        
        # Análisis por producto
        for product_id in reviews_df['product_id'].unique():
            product_analysis = self.analyze_product_sentiment(product_id, reviews_df)
            insights[product_id] = product_analysis
        
        # Ranking de productos por sentimiento
        product_scores = [(pid, data['average_score']) for pid, data in insights.items()]
        product_scores.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'product_insights': insights,
            'top_products': product_scores[:5],
            'worst_products': product_scores[-5:],
            'overall_stats': {
                'total_products_analyzed': len(insights),
                'avg_sentiment_score': np.mean([data['average_score'] for data in insights.values()])
            }
        }

# Crear sistema de análisis
sentiment_analyzer = SentimentAnalysisSystem(best_model)

# Ejemplos de uso
# Analizar texto individual
sample_text = "Este producto es excelente, muy buena calidad y llegó rápido"
result = sentiment_analyzer.analyze_sentiment(sample_text)
print(f"Texto: {sample_text}")
print(f"Sentimiento: {result['sentiment']} (confianza: {result['confidence']:.2f})")

# Analizar producto específico
product_analysis = sentiment_analyzer.analyze_product_sentiment('PROD_001', reviews)
print(f"\nAnálisis del producto PROD_001:")
print(f"Sentimiento general: {product_analysis['overall_sentiment']}")
print(f"Total reseñas: {product_analysis['total_reviews']}")
print(f"% Positivas: {product_analysis['positive_percentage']:.1f}%")
print(f"% Negativas: {product_analysis['negative_percentage']:.1f}%")

# Insights generales
insights = sentiment_analyzer.get_sentiment_insights(reviews)
print(f"\nTop 3 productos mejor valorados:")
for i, (product_id, score) in enumerate(insights['top_products'][:3], 1):
    print(f"{i}. {product_id}: {score:.2f}")
```

### Guardado y Despliegue
```python
import joblib
import pickle

# Guardar modelo entrenado
joblib.dump(best_model, 'sentiment_model.pkl')

# Guardar sistema completo
with open('sentiment_analyzer.pkl', 'wb') as f:
    pickle.dump(sentiment_analyzer, f)

print("Modelos guardados exitosamente")

# Función para cargar y usar en producción
def load_sentiment_system():
    with open('sentiment_analyzer.pkl', 'rb') as f:
        return pickle.load(f)

# API simple para el modelo
from flask import Flask, request, jsonify

app = Flask(__name__)
sentiment_system = load_sentiment_system()

@app.route('/analyze_sentiment', methods=['POST'])
def api_analyze_sentiment():
    data = request.json
    text = data.get('text', '')
    
    result = sentiment_system.analyze_sentiment(text)
    
    return jsonify(result)

@app.route('/analyze_product', methods=['POST'])
def api_analyze_product():
    data = request.json
    product_id = data.get('product_id')
    
    # Cargar reseñas del producto (en producción vendría de base de datos)
    product_analysis = sentiment_system.analyze_product_sentiment(product_id, reviews)
    
    return jsonify(product_analysis)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

Estos ejemplos demuestran implementaciones completas de sistemas de ML reales, incluyendo preparación de datos, entrenamiento, evaluación y despliegue. Cada ejemplo puede adaptarse y escalarse según las necesidades específicas del proyecto.

---
layout: ../../layouts/DocLayout.astro
title: "Introducción al Procesamiento de Lenguaje Natural"
description: "Fundamentos del NLP y técnicas modernas"
currentPath: "/nlp/introduccion"
---

# Introducción al Procesamiento de Lenguaje Natural (NLP)

El Procesamiento de Lenguaje Natural (NLP) es una rama de la inteligencia artificial que se enfoca en la interacción entre computadoras y lenguaje humano, permitiendo que las máquinas comprendan, interpreten y generen texto de manera significativa.

## ¿Qué es el NLP?

### Definición
El NLP combina lingüística computacional con modelos estadísticos, machine learning y deep learning para procesar y analizar grandes cantidades de datos de lenguaje natural.

### Objetivos Principales
- **Comprensión**: Entender el significado del texto
- **Generación**: Crear texto coherente y relevante
- **Traducción**: Convertir entre idiomas
- **Extracción**: Identificar información específica
- **Clasificación**: Categorizar texto por contenido o intención

## Desafíos del Lenguaje Natural

### 1. Ambigüedad
```python
# Ejemplos de ambigüedad en español
ejemplos_ambiguedad = [
    {
        "texto": "Banco de pescados",
        "interpretaciones": [
            "Institución financiera que vende pescados",
            "Grupo/cardumen de pescados"
        ]
    },
    {
        "texto": "Vi al hombre con el telescopio",
        "interpretaciones": [
            "Usé un telescopio para ver al hombre",
            "Vi al hombre que tenía un telescopio"
        ]
    }
]

def detectar_ambiguedad(texto):
    """Función simplificada para detectar posibles ambigüedades"""
    
    # Palabras polisémicas comunes
    palabras_polisemicas = {
        'banco': ['institución financiera', 'asiento', 'grupo de peces'],
        'gato': ['animal', 'herramienta mecánica'],
        'lima': ['fruta', 'ciudad', 'herramienta'],
        'cabo': ['extremo', 'rango militar', 'accidente geográfico']
    }
    
    palabras = texto.lower().split()
    ambigüedades = []
    
    for palabra in palabras:
        if palabra in palabras_polisemicas:
            ambigüedades.append({
                'palabra': palabra,
                'significados': palabras_polisemicas[palabra]
            })
    
    return ambigüedades
```

### 2. Variabilidad Lingüística
```python
# Diferentes formas de expresar la misma idea
expresiones_equivalentes = [
    "Hace mucho calor",
    "Está muy caluroso",
    "La temperatura es elevada",
    "Hace un calor tremendo",
    "Está que arde"
]

def normalizar_expresiones(texto):
    """Normalizar expresiones similares"""
    
    patrones_normalizacion = {
        r'hace (mucho )?calor|está (muy )?caluroso|temperatura elevada': 'clima_calido',
        r'hace (mucho )?frío|está (muy )?fresco|temperatura baja': 'clima_frio',
        r'llueve|está lloviendo|caen gotas': 'precipitacion'
    }
    
    import re
    
    texto_normalizado = texto.lower()
    for patron, normalizacion in patrones_normalizacion.items():
        texto_normalizado = re.sub(patron, normalizacion, texto_normalizado)
    
    return texto_normalizado
```

## Pipeline de NLP

### 1. Preprocesamiento de Texto
```python
import re
import string
from collections import Counter

class TextPreprocessor:
    def __init__(self, language='es'):
        self.language = language
        self.stop_words = self.load_stop_words()
    
    def load_stop_words(self):
        """Cargar palabras vacías en español"""
        stop_words_es = {
            'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 
            'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 
            'para', 'al', 'del', 'los', 'las', 'pero', 'sus', 'fue',
            'ser', 'tiene', 'ya', 'todo', 'esta', 'muy', 'puede',
            'como', 'sin', 'más', 'está', 'o', 'si', 'me', 'han'
        }
        return stop_words_es
    
    def clean_text(self, text):
        """Limpiar texto básico"""
        # Convertir a minúsculas
        text = text.lower()
        
        # Remover URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remover menciones y hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remover números
        text = re.sub(r'\d+', '', text)
        
        # Remover puntuación
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remover espacios extra
        text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, text):
        """Tokenizar texto"""
        return text.split()
    
    def remove_stop_words(self, tokens):
        """Remover palabras vacías"""
        return [token for token in tokens if token not in self.stop_words]
    
    def stem_words(self, tokens):
        """Stemming simple para español"""
        # Reglas básicas de stemming para español
        suffixes = ['ando', 'iendo', 'ado', 'ido', 'ar', 'er', 'ir', 'ción', 'sión']
        
        stemmed = []
        for token in tokens:
            for suffix in suffixes:
                if token.endswith(suffix) and len(token) > len(suffix) + 2:
                    token = token[:-len(suffix)]
                    break
            stemmed.append(token)
        
        return stemmed
    
    def preprocess(self, text):
        """Pipeline completo de preprocesamiento"""
        # Limpiar
        text = self.clean_text(text)
        
        # Tokenizar
        tokens = self.tokenize(text)
        
        # Remover stop words
        tokens = self.remove_stop_words(tokens)
        
        # Stemming
        tokens = self.stem_words(tokens)
        
        # Filtrar tokens muy cortos
        tokens = [token for token in tokens if len(token) > 2]
        
        return tokens

# Ejemplo de uso
preprocessor = TextPreprocessor()
texto = "El procesamiento de lenguaje natural es una rama fascinante de la IA que está revolucionando la forma en que interactuamos con las computadoras."
tokens = preprocessor.preprocess(texto)
print(f"Tokens procesados: {tokens}")
```

### 2. Extracción de Características
```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np

class FeatureExtractor:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
    
    def extract_tfidf_features(self, documents, max_features=1000):
        """Extraer características TF-IDF"""
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # Unigramas y bigramas
            stop_words=None  # Ya removimos stop words
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        return tfidf_matrix, feature_names
    
    def extract_bag_of_words(self, documents, max_features=1000):
        """Extraer características Bag of Words"""
        
        self.count_vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=(1, 2)
        )
        
        bow_matrix = self.count_vectorizer.fit_transform(documents)
        feature_names = self.count_vectorizer.get_feature_names_out()
        
        return bow_matrix, feature_names
    
    def extract_statistical_features(self, text):
        """Extraer características estadísticas del texto"""
        
        features = {}
        
        # Longitud del texto
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(text.split('.'))
        
        # Promedio de palabras por oración
        sentences = text.split('.')
        if len(sentences) > 0:
            features['avg_words_per_sentence'] = sum(len(s.split()) for s in sentences) / len(sentences)
        else:
            features['avg_words_per_sentence'] = 0
        
        # Longitud promedio de palabras
        words = text.split()
        if len(words) > 0:
            features['avg_word_length'] = sum(len(word) for word in words) / len(words)
        else:
            features['avg_word_length'] = 0
        
        # Proporción de palabras únicas
        unique_words = set(words)
        features['unique_word_ratio'] = len(unique_words) / len(words) if len(words) > 0 else 0
        
        # Conteo de signos de puntuación
        features['punctuation_count'] = sum(1 for char in text if char in string.punctuation)
        
        return features
    
    def extract_readability_features(self, text):
        """Extraer características de legibilidad"""
        
        # Índice de legibilidad simplificado
        words = text.split()
        sentences = text.split('.')
        
        if len(sentences) == 0 or len(words) == 0:
            return {'flesch_score': 0, 'complexity': 'unknown'}
        
        avg_sentence_length = len(words) / len(sentences)
        
        # Contar sílabas aproximadamente (vocal groups)
        def count_syllables(word):
            vowels = 'aeiouáéíóú'
            count = 0
            prev_was_vowel = False
            
            for char in word.lower():
                if char in vowels:
                    if not prev_was_vowel:
                        count += 1
                    prev_was_vowel = True
                else:
                    prev_was_vowel = False
            
            return max(1, count)  # Al menos 1 sílaba por palabra
        
        total_syllables = sum(count_syllables(word) for word in words)
        avg_syllables_per_word = total_syllables / len(words)
        
        # Fórmula simplificada de legibilidad
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        # Clasificar complejidad
        if flesch_score >= 90:
            complexity = 'muy_facil'
        elif flesch_score >= 80:
            complexity = 'facil'
        elif flesch_score >= 70:
            complexity = 'moderado'
        elif flesch_score >= 60:
            complexity = 'dificil'
        else:
            complexity = 'muy_dificil'
        
        return {
            'flesch_score': flesch_score,
            'complexity': complexity,
            'avg_sentence_length': avg_sentence_length,
            'avg_syllables_per_word': avg_syllables_per_word
        }

# Ejemplo de uso
extractor = FeatureExtractor()

documentos = [
    "El procesamiento de lenguaje natural es fascinante",
    "La inteligencia artificial está revolucionando el mundo",
    "Los algoritmos de machine learning son muy poderosos"
]

# Procesar documentos
preprocessor = TextPreprocessor()
docs_procesados = [' '.join(preprocessor.preprocess(doc)) for doc in documentos]

# Extraer características TF-IDF
tfidf_matrix, feature_names = extractor.extract_tfidf_features(docs_procesados)
print(f"Forma de matriz TF-IDF: {tfidf_matrix.shape}")
print(f"Primeras 10 características: {feature_names[:10]}")
```

## Tareas Fundamentales de NLP

### 1. Análisis de Sentimientos
```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class SentimentAnalyzer:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.preprocessor = TextPreprocessor()
    
    def prepare_data(self, texts, labels):
        """Preparar datos para entrenamiento"""
        
        # Preprocesar textos
        processed_texts = []
        for text in texts:
            tokens = self.preprocessor.preprocess(text)
            processed_texts.append(' '.join(tokens))
        
        return processed_texts, labels
    
    def train(self, texts, labels):
        """Entrenar modelo de análisis de sentimientos"""
        
        # Preparar datos
        processed_texts, labels = self.prepare_data(texts, labels)
        
        # Vectorizar
        self.vectorizer = TfidfVectorizer(max_features=5000)
        X = self.vectorizer.fit_transform(processed_texts)
        
        # Entrenar modelo
        self.model = LogisticRegression(random_state=42)
        self.model.fit(X, labels)
        
        return self
    
    def predict(self, text):
        """Predecir sentimiento de un texto"""
        
        if self.model is None or self.vectorizer is None:
            raise ValueError("Modelo no entrenado")
        
        # Preprocesar
        tokens = self.preprocessor.preprocess(text)
        processed_text = ' '.join(tokens)
        
        # Vectorizar
        X = self.vectorizer.transform([processed_text])
        
        # Predecir
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0].max()
        
        return prediction, probability
    
    def get_feature_importance(self, top_n=20):
        """Obtener características más importantes"""
        
        if self.model is None or self.vectorizer is None:
            raise ValueError("Modelo no entrenado")
        
        feature_names = self.vectorizer.get_feature_names_out()
        coefficients = self.model.coef_[0]
        
        # Obtener características más positivas y negativas
        feature_importance = list(zip(feature_names, coefficients))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'most_positive': feature_importance[:top_n],
            'most_negative': feature_importance[-top_n:]
        }

# Ejemplo de datos de entrenamiento
datos_entrenamiento = [
    ("Me encanta este producto, es fantástico", "positivo"),
    ("Terrible servicio, muy decepcionante", "negativo"),
    ("El producto está bien, nada especial", "neutral"),
    ("Excelente calidad y precio", "positivo"),
    ("No recomiendo para nada", "negativo"),
    ("Cumple con las expectativas", "neutral")
]

texts, labels = zip(*datos_entrenamiento)

# Entrenar analizador
analyzer = SentimentAnalyzer()
analyzer.train(texts, labels)

# Predecir nuevo texto
nuevo_texto = "Estoy muy contento con mi compra"
sentimiento, confianza = analyzer.predict(nuevo_texto)
print(f"Sentimiento: {sentimiento}, Confianza: {confianza:.2f}")
```

### 2. Reconocimiento de Entidades Nombradas (NER)
```python
import re

class SimpleNER:
    def __init__(self):
        self.patterns = self.create_patterns()
    
    def create_patterns(self):
        """Crear patrones para reconocer entidades"""
        
        patterns = {
            'PERSONA': [
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Nombre Apellido
                r'\b(?:Sr\.|Sra\.|Dr\.|Dra\.)\s+[A-Z][a-z]+\b'  # Títulos
            ],
            'LUGAR': [
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:ciudad|Ciudad|provincia|Provincia)\b',
                r'\b(?:Madrid|Barcelona|Sevilla|Valencia|Bilbao)\b'
            ],
            'ORGANIZACION': [
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:S\.A\.|S\.L\.|Inc\.|Corp\.)\b',
                r'\b(?:Universidad|Instituto|Ministerio)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
            ],
            'FECHA': [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                r'\b\d{1,2}\s+de\s+(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\s+de\s+\d{4}\b'
            ],
            'DINERO': [
                r'\b\d+(?:\.\d{3})*(?:,\d{2})?\s*(?:€|euros?|USD|dólares?)\b',
                r'\b(?:€|USD)\s*\d+(?:\.\d{3})*(?:,\d{2})?\b'
            ],
            'EMAIL': [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            'TELEFONO': [
                r'\b(?:\+34\s*)?[6789]\d{8}\b',
                r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{3}\b'
            ]
        }
        
        return patterns
    
    def extract_entities(self, text):
        """Extraer entidades del texto"""
        
        entities = []
        
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    entities.append({
                        'text': match.group(),
                        'label': entity_type,
                        'start': match.start(),
                        'end': match.end()
                    })
        
        # Ordenar por posición
        entities.sort(key=lambda x: x['start'])
        
        return entities
    
    def annotate_text(self, text):
        """Anotar texto con entidades encontradas"""
        
        entities = self.extract_entities(text)
        
        # Crear texto anotado
        annotated = text
        offset = 0
        
        for entity in entities:
            start = entity['start'] + offset
            end = entity['end'] + offset
            
            annotation = f"[{entity['text']}]({entity['label']})"
            
            annotated = annotated[:start] + annotation + annotated[end:]
            offset += len(annotation) - len(entity['text'])
        
        return annotated, entities

# Ejemplo de uso
ner = SimpleNER()

texto = """
Juan Pérez, director de TechCorp S.A., anunció que la empresa ubicada en Madrid 
invertirá 2.5 millones de euros en nuevas tecnologías. El anuncio se realizó 
el 15 de marzo de 2024. Para más información, contacte a info@techcorp.com 
o llame al 91 234 5678.
"""

texto_anotado, entidades = ner.annotate_text(texto)

print("Entidades encontradas:")
for entidad in entidades:
    print(f"- {entidad['text']} ({entidad['label']})")

print(f"\nTexto anotado:\n{texto_anotado}")
```

### 3. Clasificación de Texto
```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

class TextClassifier:
    def __init__(self, algorithm='logistic'):
        self.algorithm = algorithm
        self.pipeline = None
        self.classes = None
    
    def create_pipeline(self):
        """Crear pipeline de clasificación"""
        
        if self.algorithm == 'logistic':
            classifier = LogisticRegression(random_state=42)
        elif self.algorithm == 'random_forest':
            classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.algorithm == 'svm':
            classifier = SVC(kernel='linear', random_state=42)
        else:
            raise ValueError(f"Algoritmo no soportado: {self.algorithm}")
        
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('classifier', classifier)
        ])
        
        return pipeline
    
    def train(self, texts, labels):
        """Entrenar clasificador"""
        
        # Preprocesar textos
        preprocessor = TextPreprocessor()
        processed_texts = []
        
        for text in texts:
            tokens = preprocessor.preprocess(text)
            processed_texts.append(' '.join(tokens))
        
        # Crear y entrenar pipeline
        self.pipeline = self.create_pipeline()
        self.pipeline.fit(processed_texts, labels)
        
        # Guardar clases
        self.classes = list(set(labels))
        
        return self
    
    def predict(self, text):
        """Clasificar texto"""
        
        if self.pipeline is None:
            raise ValueError("Modelo no entrenado")
        
        # Preprocesar
        preprocessor = TextPreprocessor()
        tokens = preprocessor.preprocess(text)
        processed_text = ' '.join(tokens)
        
        # Predecir
        prediction = self.pipeline.predict([processed_text])[0]
        
        # Obtener probabilidades si están disponibles
        try:
            probabilities = self.pipeline.predict_proba([processed_text])[0]
            confidence = max(probabilities)
        except AttributeError:
            confidence = 1.0
        
        return prediction, confidence
    
    def evaluate(self, texts, labels):
        """Evaluar modelo"""
        
        if self.pipeline is None:
            raise ValueError("Modelo no entrenado")
        
        # Preprocesar textos de prueba
        preprocessor = TextPreprocessor()
        processed_texts = []
        
        for text in texts:
            tokens = preprocessor.preprocess(text)
            processed_texts.append(' '.join(tokens))
        
        # Predecir
        predictions = self.pipeline.predict(processed_texts)
        
        # Calcular métricas
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'detailed_report': classification_report(labels, predictions)
        }

# Ejemplo: Clasificación de noticias
noticias_datos = [
    ("El Real Madrid ganó el partido por 3-1", "deportes"),
    ("La bolsa subió un 2% en la sesión de hoy", "economia"),
    ("Nueva película de Marvel rompe récords", "entretenimiento"),
    ("Descubren nueva especie en el Amazonas", "ciencia"),
    ("El gobierno anuncia nuevas medidas fiscales", "politica"),
    ("El Barcelona fichó un nuevo delantero", "deportes"),
    ("Tesla presenta su nuevo modelo eléctrico", "tecnologia")
]

texts, labels = zip(*noticias_datos)

# Entrenar clasificador
classifier = TextClassifier(algorithm='logistic')
classifier.train(texts, labels)

# Clasificar nueva noticia
nueva_noticia = "Apple lanza nueva versión de iPhone con inteligencia artificial"
categoria, confianza = classifier.predict(nueva_noticia)
print(f"Categoría: {categoria}, Confianza: {confianza:.2f}")
```

## Métricas de Evaluación en NLP

### 1. Métricas para Clasificación
```python
def evaluate_nlp_classification(y_true, y_pred, classes):
    """Evaluar clasificación de texto"""
    
    from sklearn.metrics import confusion_matrix, classification_report
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # Visualizar matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Matriz de Confusión')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.show()
    
    # Reporte detallado
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    
    return {
        'confusion_matrix': cm,
        'classification_report': report,
        'accuracy': report['accuracy'],
        'macro_avg_f1': report['macro avg']['f1-score'],
        'weighted_avg_f1': report['weighted avg']['f1-score']
    }
```

### 2. Métricas para Generación de Texto
```python
def evaluate_text_generation(generated_texts, reference_texts):
    """Evaluar calidad de generación de texto"""
    
    # BLEU Score
    from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
    
    bleu_scores = []
    for gen, ref in zip(generated_texts, reference_texts):
        bleu = sentence_bleu([ref.split()], gen.split())
        bleu_scores.append(bleu)
    
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    
    # ROUGE Score (simplificado)
    def simple_rouge_1(generated, reference):
        gen_words = set(generated.split())
        ref_words = set(reference.split())
        
        if len(ref_words) == 0:
            return 0
        
        overlap = len(gen_words.intersection(ref_words))
        precision = overlap / len(gen_words) if len(gen_words) > 0 else 0
        recall = overlap / len(ref_words)
        
        if precision + recall == 0:
            return 0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    rouge_scores = []
    for gen, ref in zip(generated_texts, reference_texts):
        rouge = simple_rouge_1(gen, ref)
        rouge_scores.append(rouge)
    
    avg_rouge = sum(rouge_scores) / len(rouge_scores)
    
    return {
        'bleu': avg_bleu,
        'rouge_1': avg_rouge,
        'individual_bleu': bleu_scores,
        'individual_rouge': rouge_scores
    }
```

El NLP es un campo en constante evolución que combina técnicas tradicionales con avances modernos de deep learning, ofreciendo soluciones cada vez más sofisticadas para entender y generar lenguaje humano.

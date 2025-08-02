---
layout: ../../layouts/DocLayout.astro
title: "Embeddings y Representaciones"
description: "Comprende qué son los embeddings y cómo representan información en espacios vectoriales"
section: "nlp"
---

# Embeddings y Representaciones

## ¿Qué son los Embeddings?

Los **embeddings** son representaciones vectoriales de palabras, frases o documentos en un espacio multidimensional continuo. Estas representaciones capturan relaciones semánticas y sintácticas de manera que palabras con significados similares tienen representaciones vectoriales cercanas.

## Tipos de Embeddings

### 1. Word Embeddings

#### Word2Vec
- **Skip-gram**: Predice palabras contextuales dada una palabra central
- **CBOW**: Predice una palabra central dado su contexto

```python
from gensim.models import Word2Vec

# Datos de entrenamiento
sentences = [['el', 'gato', 'está', 'en', 'la', 'mesa'],
             ['el', 'perro', 'corre', 'en', 'el', 'parque']]

# Entrenar modelo Word2Vec
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Obtener vector de una palabra
vector_gato = model.wv['gato']
print(f"Vector para 'gato': {vector_gato[:5]}...")

# Encontrar palabras similares
similares = model.wv.most_similar('gato', topn=3)
print(f"Palabras similares a 'gato': {similares}")
```

#### GloVe (Global Vectors)
GloVe combina las ventajas de métodos de factorización matricial y de ventana de contexto local.

```python
import numpy as np
from sklearn.decomposition import TruncatedSVD

# Simulación de matriz de co-ocurrencia
def crear_matriz_coocurrencia(corpus, vocab, ventana=2):
    vocab_to_idx = {word: i for i, word in enumerate(vocab)}
    matriz = np.zeros((len(vocab), len(vocab)))
    
    for frase in corpus:
        for i, palabra in enumerate(frase):
            if palabra in vocab_to_idx:
                idx_palabra = vocab_to_idx[palabra]
                # Ventana de contexto
                inicio = max(0, i - ventana)
                fin = min(len(frase), i + ventana + 1)
                
                for j in range(inicio, fin):
                    if i != j and frase[j] in vocab_to_idx:
                        idx_contexto = vocab_to_idx[frase[j]]
                        matriz[idx_palabra][idx_contexto] += 1
    
    return matriz

# Ejemplo de uso
corpus = [['el', 'gato', 'duerme'], ['el', 'perro', 'corre']]
vocab = ['el', 'gato', 'duerme', 'perro', 'corre']
matriz_cooc = crear_matriz_coocurrencia(corpus, vocab)
print("Matriz de co-ocurrencia:")
print(matriz_cooc)
```

### 2. Sentence Embeddings

#### BERT (Bidirectional Encoder Representations from Transformers)

```python
from transformers import BertTokenizer, BertModel
import torch

# Cargar modelo preentrenado
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def obtener_embedding_frase(frase):
    # Tokenizar
    inputs = tokenizer(frase, return_tensors='pt', padding=True, truncation=True)
    
    # Obtener embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Usar el token [CLS] como representación de la frase
    embedding_frase = outputs.last_hidden_state[:, 0, :]
    return embedding_frase

# Ejemplo
frase = "El gato está durmiendo"
embedding = obtener_embedding_frase(frase)
print(f"Dimensión del embedding: {embedding.shape}")
```

#### Sentence-BERT

```python
from sentence_transformers import SentenceTransformer

# Cargar modelo
model = SentenceTransformer('all-MiniLM-L6-v2')

# Frases de ejemplo
frases = [
    "El gato está durmiendo en el sofá",
    "Un felino descansa en el mueble",
    "El perro corre en el parque",
    "Python es un lenguaje de programación"
]

# Obtener embeddings
embeddings = model.encode(frases)

# Calcular similitudes
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

similitudes = cosine_similarity(embeddings)
print("Matriz de similitudes:")
for i, frase in enumerate(frases):
    print(f"{i}: {frase}")

print("\nSimilitudes coseno:")
print(np.round(similitudes, 3))
```

### 3. Document Embeddings

#### Doc2Vec

```python
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

# Preparar documentos
documentos = [
    "El aprendizaje automático es una rama de la inteligencia artificial",
    "Las redes neuronales son fundamentales en deep learning",
    "El procesamiento de lenguaje natural trabaja con texto"
]

# Crear documentos etiquetados
tagged_docs = [TaggedDocument(words=doc.split(), tags=[i]) 
               for i, doc in enumerate(documentos)]

# Entrenar modelo
model = Doc2Vec(tagged_docs, vector_size=50, window=2, min_count=1, workers=4)

# Obtener embedding de documento
doc_embedding = model.infer_vector("machine learning artificial intelligence".split())
print(f"Embedding del documento: {doc_embedding[:5]}...")

# Encontrar documentos similares
similares = model.dv.most_similar([doc_embedding], topn=2)
print(f"Documentos similares: {similares}")
```

## Técnicas Avanzadas

### 1. Embeddings Contextualizados

A diferencia de los embeddings estáticos, los contextualizados cambian según el contexto:

```python
from transformers import AutoTokenizer, AutoModel
import torch

def embeddings_contextualizados(frases, modelo_nombre='bert-base-uncased'):
    tokenizer = AutoTokenizer.from_pretrained(modelo_nombre)
    model = AutoModel.from_pretrained(modelo_nombre)
    
    embeddings = []
    
    for frase in frases:
        inputs = tokenizer(frase, return_tensors='pt', padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Promediar todos los tokens (excepto [CLS] y [SEP])
        embedding = outputs.last_hidden_state.mean(dim=1)
        embeddings.append(embedding)
    
    return torch.cat(embeddings, dim=0)

# Ejemplo con palabras polisémicas
frases = [
    "El banco está lleno de peces",  # banco = orilla del río
    "Voy al banco a sacar dinero"    # banco = institución financiera
]

embeddings = embeddings_contextualizados(frases)
print(f"Embeddings contextualizados shape: {embeddings.shape}")

# Calcular similitud
similitud = torch.cosine_similarity(embeddings[0], embeddings[1], dim=0)
print(f"Similitud entre los dos usos de 'banco': {similitud.item():.3f}")
```

### 2. Fine-tuning de Embeddings

```python
from transformers import AutoTokenizer, AutoModel, AdamW
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

class ClasificadorConEmbeddings(nn.Module):
    def __init__(self, modelo_base, num_clases):
        super().__init__()
        self.bert = AutoModel.from_pretrained(modelo_base)
        self.clasificador = nn.Linear(self.bert.config.hidden_size, num_clases)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.clasificador(output)

# Ejemplo de uso
modelo = ClasificadorConEmbeddings('bert-base-uncased', num_clases=3)
optimizer = AdamW(modelo.parameters(), lr=2e-5)
```

## Evaluación de Embeddings

### 1. Evaluación Intrínseca

```python
def evaluar_analogias(model, analogias):
    """
    Evalúa embeddings usando analogías como "rey - hombre + mujer = reina"
    """
    correctos = 0
    total = 0
    
    for a, b, c, d_esperado in analogias:
        try:
            # Calcular d = b - a + c
            resultado = model.wv.most_similar(
                positive=[b, c], negative=[a], topn=1
            )[0][0]
            
            if resultado == d_esperado:
                correctos += 1
            total += 1
            
        except KeyError:
            continue
    
    return correctos / total if total > 0 else 0

# Ejemplos de analogías
analogias = [
    ('hombre', 'rey', 'mujer', 'reina'),
    ('madrid', 'españa', 'paris', 'francia')
]
```

### 2. Evaluación Extrínseca

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def evaluar_clasificacion(embeddings, etiquetas):
    """
    Evalúa embeddings en una tarea de clasificación
    """
    # Dividir en entrenamiento y prueba
    split = int(0.8 * len(embeddings))
    X_train, X_test = embeddings[:split], embeddings[split:]
    y_train, y_test = etiquetas[:split], etiquetas[split:]
    
    # Entrenar clasificador
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    
    # Evaluar
    predicciones = clf.predict(X_test)
    return accuracy_score(y_test, predicciones)
```

## Aplicaciones Prácticas

### 1. Búsqueda Semántica

```python
import faiss
import numpy as np

class BuscadorSemantico:
    def __init__(self, modelo_embeddings):
        self.modelo = modelo_embeddings
        self.documentos = []
        self.index = None
    
    def indexar_documentos(self, documentos):
        self.documentos = documentos
        embeddings = self.modelo.encode(documentos)
        
        # Crear índice FAISS
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Producto interno
        
        # Normalizar para usar similitud coseno
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
    
    def buscar(self, consulta, k=5):
        embedding_consulta = self.modelo.encode([consulta])
        faiss.normalize_L2(embedding_consulta)
        
        scores, indices = self.index.search(embedding_consulta.astype('float32'), k)
        
        resultados = []
        for score, idx in zip(scores[0], indices[0]):
            resultados.append({
                'documento': self.documentos[idx],
                'score': float(score)
            })
        
        return resultados

# Ejemplo de uso
buscador = BuscadorSemantico(model)  # usando el modelo de sentence-transformers
documentos = [
    "Python es un lenguaje de programación",
    "El machine learning utiliza algoritmos",
    "Las redes neuronales imitan el cerebro"
]

buscador.indexar_documentos(documentos)
resultados = buscador.buscar("algoritmos de IA", k=2)
for resultado in resultados:
    print(f"Score: {resultado['score']:.3f} - {resultado['documento']}")
```

### 2. Clustering de Documentos

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def clustering_documentos(documentos, n_clusters=3):
    # Obtener embeddings
    embeddings = model.encode(documentos)
    
    # Aplicar K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    
    # Visualizar con PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=clusters, cmap='viridis')
    plt.colorbar(scatter)
    plt.title('Clustering de Documentos')
    
    # Anotar algunos puntos
    for i, doc in enumerate(documentos[:5]):  # Solo primeros 5
        plt.annotate(doc[:30] + '...', 
                    (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    return clusters

# Ejemplo
documentos_ejemplo = [
    "El deep learning es una técnica de IA",
    "Los gatos son animales domésticos",
    "Python es útil para data science",
    "Los perros son mascotas leales",
    "TensorFlow es un framework de ML"
]

clusters = clustering_documentos(documentos_ejemplo)
```

## Mejores Prácticas

### 1. Selección de Embeddings
- **Tamaño del corpus**: Word2Vec para corpus grandes, embeddings preentrenados para corpus pequeños
- **Dominio específico**: Fine-tuning de embeddings preentrenados
- **Tarea específica**: Embeddings contextualizados para tareas complejas

### 2. Preprocesamiento
```python
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocesar_texto(texto):
    # Convertir a minúsculas
    texto = texto.lower()
    
    # Remover caracteres especiales
    texto = re.sub(r'[^a-záéíóúñ\s]', '', texto)
    
    # Tokenizar
    tokens = word_tokenize(texto)
    
    # Remover stopwords
    stop_words = set(stopwords.words('spanish'))
    tokens = [token for token in tokens if token not in stop_words]
    
    return tokens

# Ejemplo
texto = "¡Hola! Este es un ejemplo de texto para procesar."
tokens = preprocesar_texto(texto)
print(f"Tokens: {tokens}")
```

### 3. Optimización de Rendimiento
```python
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

def procesar_embeddings_paralelo(textos, modelo, batch_size=32):
    """
    Procesa embeddings en paralelo para mejorar rendimiento
    """
    def procesar_batch(batch):
        return modelo.encode(batch)
    
    # Dividir en batches
    batches = [textos[i:i+batch_size] for i in range(0, len(textos), batch_size)]
    
    # Procesar en paralelo
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        resultados = list(executor.map(procesar_batch, batches))
    
    # Concatenar resultados
    return np.vstack(resultados)
```

## Recursos Adicionales

- **Bibliotecas**: Gensim, Hugging Face Transformers, Sentence Transformers
- **Datasets**: Common Crawl, Wikipedia, BookCorpus
- **Modelos preentrenados**: BERT, RoBERTa, DistilBERT, GPT
- **Herramientas de visualización**: TensorBoard, Projector, t-SNE, UMAP

Los embeddings son fundamentales en el NLP moderno, proporcionando representaciones ricas que capturan semántica y permiten transferir conocimiento entre tareas diferentes.

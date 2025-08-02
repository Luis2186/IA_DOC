---
layout: ../../layouts/DocLayout.astro
title: "Transformers: La Revolución del Deep Learning"
description: "Arquitectura Transformer y su impacto en NLP y más allá"
currentPath: "/deep-learning/transformers"
---

# Transformers: La Revolución del Deep Learning

Los Transformers han revolucionado el campo del Deep Learning, especialmente en el procesamiento de lenguaje natural, introduciendo el concepto de atención y eliminando la necesidad de recurrencia.

## ¿Qué son los Transformers?

### Concepto Fundamental
Los Transformers son una arquitectura de red neuronal que utiliza únicamente mecanismos de atención para procesar secuencias, sin recurrir a convoluciones o recurrencia.

### Ventajas Clave
- **Paralelización**: Procesamiento simultáneo de toda la secuencia
- **Atención Global**: Cada posición puede atender a todas las demás
- **Escalabilidad**: Mejor rendimiento con más datos y parámetros
- **Transfer Learning**: Excelente capacidad de transferencia

## Arquitectura del Transformer

### Componentes Principales

#### 1. Multi-Head Attention
```python
import tensorflow as tf
import numpy as np

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        
        self.dense = tf.keras.layers.Dense(d_model)
    
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        
        # Proyecciones lineales
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        # Dividir en múltiples cabezas
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # Atención escalada
        attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)
        
        # Concatenar cabezas
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        attention = tf.reshape(attention, (batch_size, -1, self.d_model))
        
        output = self.dense(attention)
        
        return output, attention_weights

def scaled_dot_product_attention(q, k, v, mask):
    """Calcular atención por producto punto escalado"""
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    
    # Escalar
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
    # Aplicar máscara
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    
    # Softmax
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    
    output = tf.matmul(attention_weights, v)
    
    return output, attention_weights
```

#### 2. Position Encoding
```python
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
    
    # Sen a índices pares
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
    # Cos a índices impares
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return tf.cast(pos_encoding, dtype=tf.float32)
```

#### 3. Feed Forward Network
```python
def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])
```

#### 4. Encoder Layer
```python
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
    
    def call(self, x, training, mask):
        # Multi-head attention
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # Residual connection
        
        # Feed forward
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # Residual connection
        
        return out2
```

#### 5. Encoder Completo
```python
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)
        
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                          for _ in range(num_layers)]
        
        self.dropout = tf.keras.layers.Dropout(rate)
    
    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        
        # Embedding + position encoding
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        
        x = self.dropout(x, training=training)
        
        # Pasar por todas las capas encoder
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        
        return x
```

## Variantes de Transformers

### 1. BERT (Bidirectional Encoder Representations from Transformers)

#### Características
- **Bidireccional**: Procesa contexto en ambas direcciones
- **Masked Language Model**: Predice palabras enmascaradas
- **Next Sentence Prediction**: Predice si dos oraciones son consecutivas

#### Implementación Simplificada
```python
class BERTModel(tf.keras.Model):
    def __init__(self, vocab_size, d_model=768, num_heads=12, num_layers=12, dff=3072):
        super(BERTModel, self).__init__()
        
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                              vocab_size, maximum_position_encoding=512)
        
        # Para Masked Language Model
        self.mlm_head = tf.keras.layers.Dense(vocab_size)
        
        # Para Next Sentence Prediction
        self.nsp_head = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model, activation='tanh'),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
    
    def call(self, inputs, training=False):
        # Encoding
        encoded = self.encoder(inputs, training=training, mask=None)
        
        # MLM predictions
        mlm_output = self.mlm_head(encoded)
        
        # NSP prediction (usando [CLS] token)
        cls_output = encoded[:, 0, :]  # [CLS] token
        nsp_output = self.nsp_head(cls_output)
        
        return mlm_output, nsp_output

# Uso para fine-tuning
def create_bert_classifier(bert_model, num_classes):
    inputs = tf.keras.Input(shape=(None,), dtype=tf.int32)
    
    # BERT encoding
    encoded = bert_model.encoder(inputs, training=False, mask=None)
    
    # Clasificación usando [CLS] token
    cls_output = encoded[:, 0, :]
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(cls_output)
    
    return tf.keras.Model(inputs, outputs)
```

### 2. GPT (Generative Pre-trained Transformer)

#### Características
- **Autoregresivo**: Genera texto de izquierda a derecha
- **Decoder-only**: Solo utiliza la parte decoder del Transformer
- **Causal Attention**: Cada posición solo puede atender a posiciones anteriores

#### Implementación Simplificada
```python
class GPTModel(tf.keras.Model):
    def __init__(self, vocab_size, d_model=768, num_heads=12, num_layers=12, dff=3072, max_seq_len=1024):
        super(GPTModel, self).__init__()
        
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_seq_len, d_model)
        
        self.decoder_layers = [DecoderLayer(d_model, num_heads, dff) 
                              for _ in range(num_layers)]
        
        self.final_layer = tf.keras.layers.Dense(vocab_size)
    
    def create_look_ahead_mask(self, size):
        """Crear máscara causal"""
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)
    
    def call(self, inputs, training=False):
        seq_len = tf.shape(inputs)[1]
        
        # Embedding + position encoding
        x = self.embedding(inputs)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        
        # Máscara causal
        look_ahead_mask = self.create_look_ahead_mask(seq_len)
        
        # Pasar por capas decoder
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, training, look_ahead_mask)
        
        # Proyección final
        output = self.final_layer(x)
        
        return output

def generate_text(model, start_string, num_generate=100, temperature=1.0):
    """Generar texto usando el modelo GPT"""
    # Convertir string inicial a tokens
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    
    text_generated = []
    
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        
        # Usar temperatura para controlar aleatoriedad
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        
        # Añadir predicción al input
        input_eval = tf.expand_dims([predicted_id], 0)
        
        text_generated.append(idx2char[predicted_id])
    
    return start_string + ''.join(text_generated)
```

### 3. T5 (Text-to-Text Transfer Transformer)

#### Características
- **Text-to-Text**: Todas las tareas como generación de texto
- **Encoder-Decoder**: Arquitectura completa del Transformer
- **Prefijos de tarea**: Identifica la tarea con prefijos

```python
class T5Model(tf.keras.Model):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, dff=2048):
        super(T5Model, self).__init__()
        
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                              vocab_size, maximum_position_encoding=512)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
                              vocab_size, maximum_position_encoding=512)
        
        self.final_layer = tf.keras.layers.Dense(vocab_size)
    
    def call(self, inp, tar, training=False):
        # Encoding
        enc_output = self.encoder(inp, training, None)
        
        # Decoding
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, None, None)
        
        final_output = self.final_layer(dec_output)
        
        return final_output, attention_weights

# Ejemplo de uso para diferentes tareas
def create_task_examples():
    examples = {
        'translation': {
            'input': 'translate English to Spanish: The house is beautiful',
            'target': 'La casa es hermosa'
        },
        'summarization': {
            'input': 'summarize: [long text here]',
            'target': '[summary here]'
        },
        'question_answering': {
            'input': 'question: What is the capital of France? context: [context]',
            'target': 'Paris'
        }
    }
    return examples
```

## Optimizaciones y Mejoras

### 1. Attention Patterns Eficientes

#### Sparse Attention
```python
class SparseAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, sparsity_pattern='local'):
        super(SparseAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.sparsity_pattern = sparsity_pattern
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)
    
    def create_sparse_mask(self, seq_len):
        if self.sparsity_pattern == 'local':
            # Atención local (ventana de 128)
            window_size = 128
            mask = tf.zeros((seq_len, seq_len))
            
            for i in range(seq_len):
                start = max(0, i - window_size // 2)
                end = min(seq_len, i + window_size // 2)
                mask = tf.tensor_scatter_nd_update(
                    mask, [[i, j] for j in range(start, end)], 
                    [1] * (end - start)
                )
            return mask
        
        return tf.ones((seq_len, seq_len))
```

#### Linear Attention
```python
class LinearAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(LinearAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)
    
    def elu_feature_map(self, x):
        return tf.nn.elu(x) + 1
    
    def call(self, q, k, v):
        # Aplicar feature map
        q = self.elu_feature_map(q)
        k = self.elu_feature_map(k)
        
        # Computación lineal O(n)
        kv = tf.einsum('...nd,...ne->...de', k, v)
        z = tf.einsum('...nd,...d->...n', q, tf.reduce_sum(k, axis=-2))
        
        out = tf.einsum('...nd,...de->...ne', q, kv) / (z[..., None] + 1e-6)
        
        return self.dense(out)
```

### 2. Techniques de Entrenamiento

#### Gradient Checkpointing
```python
@tf.recompute_grad
def transformer_layer_with_checkpointing(x, layer):
    return layer(x)

# Uso en el modelo
class MemoryEfficientTransformer(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.layers = [TransformerLayer() for _ in range(num_layers)]
    
    def call(self, x):
        for layer in self.layers:
            x = transformer_layer_with_checkpointing(x, layer)
        return x
```

#### Mixed Precision Training
```python
# Configurar política de precisión mixta
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# En las capas finales usar float32
final_layer = tf.keras.layers.Dense(
    vocab_size, 
    dtype='float32'  # Evitar underflow en softmax
)
```

### 3. Técnicas de Fine-tuning

#### LoRA (Low-Rank Adaptation)
```python
class LoRALayer(tf.keras.layers.Layer):
    def __init__(self, original_layer, rank=8, alpha=16):
        super(LoRALayer, self).__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        
        # Matrices de bajo rango
        self.lora_A = tf.Variable(
            tf.random.normal((original_layer.units, rank)) * 0.01,
            trainable=True
        )
        self.lora_B = tf.Variable(
            tf.zeros((rank, original_layer.units)),
            trainable=True
        )
        
        # Congelar capa original
        self.original_layer.trainable = False
    
    def call(self, x):
        # Salida original
        original_output = self.original_layer(x)
        
        # Adaptación LoRA
        lora_output = tf.matmul(tf.matmul(x, self.lora_A), self.lora_B)
        lora_output *= self.alpha / self.rank
        
        return original_output + lora_output
```

## Aplicaciones Modernas

### 1. Modelos de Lenguaje Grandes (LLMs)
- **GPT-3/4**: Generación de texto general
- **ChatGPT**: Conversación interactiva
- **Codex**: Generación de código
- **InstructGPT**: Siguiendo instrucciones

### 2. Visión por Computadora
- **Vision Transformer (ViT)**: Clasificación de imágenes
- **DETR**: Detección de objetos
- **Swin Transformer**: Ventanas deslizantes

### 3. Multimodal
- **CLIP**: Visión + lenguaje
- **DALL-E**: Generación de imágenes desde texto
- **Flamingo**: Comprensión multimodal

## Consideraciones Prácticas

### 1. Escalabilidad
```python
# Estrategias de paralelización
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = create_transformer_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
```

### 2. Optimización de Memoria
```python
# Gradient accumulation para batches grandes
def train_step_with_accumulation(model, optimizer, inputs, targets, accumulation_steps=4):
    accumulated_gradients = []
    
    for i in range(accumulation_steps):
        with tf.GradientTape() as tape:
            predictions = model(inputs[i])
            loss = compute_loss(targets[i], predictions) / accumulation_steps
        
        gradients = tape.gradient(loss, model.trainable_variables)
        if i == 0:
            accumulated_gradients = gradients
        else:
            accumulated_gradients = [
                acc_grad + grad for acc_grad, grad in zip(accumulated_gradients, gradients)
            ]
    
    optimizer.apply_gradients(zip(accumulated_gradients, model.trainable_variables))
```

### 3. Evaluación y Métricas
```python
def evaluate_language_model(model, test_dataset):
    total_loss = 0
    total_tokens = 0
    
    for batch in test_dataset:
        inputs, targets = batch
        predictions = model(inputs)
        loss = compute_loss(targets, predictions)
        
        total_loss += loss * len(inputs)
        total_tokens += tf.reduce_sum(tf.cast(targets != 0, tf.float32))
    
    perplexity = tf.exp(total_loss / total_tokens)
    return perplexity
```

Los Transformers han transformado el panorama del Deep Learning, estableciendo nuevos estándares en múltiples dominios y continuando evolucionando con nuevas optimizaciones y aplicaciones.

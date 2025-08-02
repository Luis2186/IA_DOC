---
layout: ../../layouts/DocLayout.astro
title: "Arquitecturas de Deep Learning"
description: "Principales arquitecturas de redes neuronales profundas"
currentPath: "/deep-learning/arquitecturas"
---

# Arquitecturas de Deep Learning

Las arquitecturas de Deep Learning han evolucionado significativamente, cada una diseñada para resolver tipos específicos de problemas. Esta sección explora las principales arquitecturas y sus aplicaciones.

## Redes Neuronales Feedforward (MLP)

### Descripción
Las redes neuronales multicapa (Multi-Layer Perceptron) son la forma más básica de redes neuronales profundas, donde la información fluye en una sola dirección desde la entrada hasta la salida.

### Arquitectura
```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Red neuronal feedforward simple
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())
```

### Características Clave
- **Capas densamente conectadas**: Cada neurona conectada a todas las de la capa siguiente
- **Activaciones no lineales**: ReLU, sigmoid, tanh
- **Backpropagation**: Algoritmo de entrenamiento estándar

### Aplicaciones
- Clasificación de datos tabulares
- Problemas de regresión
- Tareas de clasificación simple

## Redes Neuronales Convolucionales (CNN)

### Descripción
Especializadas en procesar datos con estructura de grilla, como imágenes. Utilizan convoluciones para detectar características locales.

### Componentes Principales

#### 1. Capas Convolucionales
```python
# Capa convolucional básica
conv_layer = layers.Conv2D(
    filters=32,        # Número de filtros
    kernel_size=(3, 3), # Tamaño del kernel
    activation='relu',
    padding='same'     # Mantener dimensiones
)
```

#### 2. Capas de Pooling
```python
# Max pooling para reducir dimensionalidad
pool_layer = layers.MaxPooling2D(
    pool_size=(2, 2),
    strides=2
)
```

#### 3. Arquitectura Completa
```python
def create_cnn_model():
    model = models.Sequential([
        # Primer bloque convolucional
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        
        # Segundo bloque convolucional
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Tercer bloque convolucional
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Capas densas
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    return model

cnn_model = create_cnn_model()
print(cnn_model.summary())
```

### Arquitecturas CNN Famosas

#### LeNet-5 (1998)
```python
def lenet5():
    model = models.Sequential([
        layers.Conv2D(6, (5, 5), activation='tanh', input_shape=(32, 32, 1)),
        layers.AveragePooling2D((2, 2)),
        layers.Conv2D(16, (5, 5), activation='tanh'),
        layers.AveragePooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(120, activation='tanh'),
        layers.Dense(84, activation='tanh'),
        layers.Dense(10, activation='softmax')
    ])
    return model
```

#### ResNet (Conexiones Residuales)
```python
def residual_block(x, filters):
    # Conexión principal
    shortcut = x
    
    # Primera convolución
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Segunda convolución
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Conexión residual
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    
    return x

def create_resnet():
    inputs = layers.Input(shape=(224, 224, 3))
    
    # Capa inicial
    x = layers.Conv2D(64, (7, 7), strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
    
    # Bloques residuales
    for i in range(3):
        x = residual_block(x, 64)
    
    # Capas finales
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(1000, activation='softmax')(x)
    
    return models.Model(inputs, outputs)
```

### Aplicaciones de CNN
- Reconocimiento de imágenes
- Detección de objetos
- Segmentación semántica
- Análisis médico de imágenes

## Redes Neuronales Recurrentes (RNN)

### Descripción
Diseñadas para procesar secuencias de datos, manteniendo información del estado anterior.

### RNN Básica
```python
# RNN simple
rnn_model = models.Sequential([
    layers.SimpleRNN(50, input_shape=(None, 1)),
    layers.Dense(1)
])
```

### Long Short-Term Memory (LSTM)
```python
def create_lstm_model(sequence_length, features):
    model = models.Sequential([
        layers.LSTM(50, return_sequences=True, input_shape=(sequence_length, features)),
        layers.Dropout(0.2),
        layers.LSTM(50, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(50),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

# Ejemplo para predicción de series temporales
lstm_model = create_lstm_model(sequence_length=60, features=1)
```

### Gated Recurrent Unit (GRU)
```python
def create_gru_model(vocab_size, embedding_dim, max_length):
    model = models.Sequential([
        layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        layers.GRU(128, dropout=0.2, recurrent_dropout=0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

### RNN Bidireccionales
```python
def create_bidirectional_model():
    model = models.Sequential([
        layers.Embedding(10000, 128),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model
```

### Aplicaciones de RNN
- Procesamiento de lenguaje natural
- Predicción de series temporales
- Reconocimiento de voz
- Traducción automática

## Autoencoders

### Descripción
Redes neuronales que aprenden representaciones comprimidas de los datos de entrada.

### Autoencoder Básico
```python
def create_autoencoder(input_dim, encoding_dim):
    # Encoder
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(encoding_dim, activation='relu')(input_layer)
    
    # Decoder
    decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)
    
    # Autoencoder completo
    autoencoder = models.Model(input_layer, decoded)
    
    # Encoder separado
    encoder = models.Model(input_layer, encoded)
    
    # Decoder separado
    encoded_input = layers.Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[-1]
    decoder = models.Model(encoded_input, decoder_layer(encoded_input))
    
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    
    return autoencoder, encoder, decoder
```

### Autoencoder Convolucional
```python
def create_conv_autoencoder():
    # Encoder
    input_img = layers.Input(shape=(28, 28, 1))
    
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    # Decoder
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = models.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    
    return autoencoder
```

### Variational Autoencoders (VAE)
```python
import tensorflow.keras.backend as K

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def create_vae(input_dim, latent_dim):
    # Encoder
    inputs = layers.Input(shape=(input_dim,))
    h = layers.Dense(512, activation='relu')(inputs)
    z_mean = layers.Dense(latent_dim)(h)
    z_log_var = layers.Dense(latent_dim)(h)
    
    # Sampling
    z = layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    
    # Decoder
    decoder_h = layers.Dense(512, activation='relu')
    decoder_mean = layers.Dense(input_dim, activation='sigmoid')
    
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)
    
    # VAE model
    vae = models.Model(inputs, x_decoded_mean)
    
    # Loss function
    reconstruction_loss = tf.keras.losses.binary_crossentropy(inputs, x_decoded_mean)
    reconstruction_loss *= input_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    
    return vae
```

### Aplicaciones de Autoencoders
- Reducción de dimensionalidad
- Detección de anomalías
- Denoising de imágenes
- Generación de datos

## Redes Generativas Adversarias (GANs)

### Descripción
Dos redes neuronales compitiendo: un generador que crea datos falsos y un discriminador que trata de detectarlos.

### GAN Básica
```python
def create_generator(latent_dim):
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_dim=latent_dim),
        layers.Dense(256, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(784, activation='tanh'),
        layers.Reshape((28, 28, 1))
    ])
    return model

def create_discriminator():
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28, 1)),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def create_gan(generator, discriminator):
    discriminator.trainable = False
    
    gan_input = layers.Input(shape=(100,))
    generated_image = generator(gan_input)
    gan_output = discriminator(generated_image)
    
    gan = models.Model(gan_input, gan_output)
    gan.compile(optimizer='adam', loss='binary_crossentropy')
    
    return gan
```

### Deep Convolutional GAN (DCGAN)
```python
def create_dcgan_generator(latent_dim):
    model = models.Sequential([
        layers.Dense(7*7*256, input_dim=latent_dim),
        layers.BatchNormalization(),
        layers.Reshape((7, 7, 256)),
        
        layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', activation='tanh')
    ])
    return model

def create_dcgan_discriminator():
    model = models.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=(28, 28, 1)),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),
        
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),
        
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model
```

### Aplicaciones de GANs
- Generación de imágenes realistas
- Transferencia de estilo
- Aumentación de datos
- Super-resolución de imágenes

## Arquitecturas Híbridas

### CNN + RNN
```python
def create_cnn_rnn_model():
    # Para análisis de video o secuencias de imágenes
    model = models.Sequential([
        layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu'), 
                              input_shape=(None, 64, 64, 3)),
        layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
        layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu')),
        layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
        layers.TimeDistributed(layers.Flatten()),
        
        layers.LSTM(50),
        layers.Dense(10, activation='softmax')
    ])
    
    return model
```

### Attention Mechanism
```python
def attention_layer(inputs, attention_dim):
    # Mecanismo de atención simple
    attention_weights = layers.Dense(attention_dim, activation='tanh')(inputs)
    attention_weights = layers.Dense(1, activation='softmax')(attention_weights)
    
    weighted_input = layers.Multiply()([inputs, attention_weights])
    return layers.GlobalSumPooling1D()(weighted_input)

def create_attention_model(vocab_size, embedding_dim, max_length):
    inputs = layers.Input(shape=(max_length,))
    x = layers.Embedding(vocab_size, embedding_dim)(inputs)
    x = layers.LSTM(128, return_sequences=True)(x)
    
    # Aplicar atención
    attended = attention_layer(x, 64)
    
    outputs = layers.Dense(1, activation='sigmoid')(attended)
    
    model = models.Model(inputs, outputs)
    return model
```

## Mejores Prácticas

### 1. Inicialización de Pesos
```python
# He initialization para ReLU
layers.Dense(64, activation='relu', 
            kernel_initializer='he_normal')

# Xavier/Glorot para tanh/sigmoid
layers.Dense(64, activation='tanh', 
            kernel_initializer='glorot_normal')
```

### 2. Normalización
```python
# Batch Normalization
model.add(layers.BatchNormalization())

# Layer Normalization
model.add(layers.LayerNormalization())
```

### 3. Regularización
```python
# Dropout
model.add(layers.Dropout(0.5))

# L1/L2 Regularization
layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.01))
```

### 4. Optimización de Hiperparámetros
```python
import keras_tuner as kt

def build_model(hp):
    model = models.Sequential()
    
    # Ajustar número de capas y neuronas
    for i in range(hp.Int('num_layers', 2, 20)):
        model.add(layers.Dense(
            units=hp.Int(f'units_{i}', 32, 512, step=32),
            activation='relu'
        ))
        model.add(layers.Dropout(hp.Float(f'dropout_{i}', 0, 0.5, step=0.1)))
    
    model.add(layers.Dense(10, activation='softmax'))
    
    model.compile(
        optimizer=hp.Choice('optimizer', ['adam', 'rmsprop']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10
)
```

## Consideraciones de Implementación

### 1. Gestión de Memoria
```python
# Usar generadores para datasets grandes
def data_generator(batch_size):
    while True:
        # Cargar y yield batch de datos
        yield batch_x, batch_y

# Mixed Precision Training
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
```

### 2. Monitoreo y Callbacks
```python
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
    tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True),
    tf.keras.callbacks.TensorBoard(log_dir='logs')
]

model.fit(x_train, y_train, callbacks=callbacks)
```

### 3. Transfer Learning
```python
# Usar modelo preentrenado
base_model = tf.keras.applications.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])
```

Cada arquitectura de Deep Learning tiene sus fortalezas específicas, y la elección correcta depende del tipo de datos, el problema a resolver y los recursos computacionales disponibles.

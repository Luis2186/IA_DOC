---
layout: ../../layouts/DocLayout.astro
title: "Desarrollo de Chatbots"
description: "Aprende a crear chatbots inteligentes con NLP y técnicas modernas de IA conversacional"
section: "nlp"
---

# Desarrollo de Chatbots

## ¿Qué es un Chatbot?

Un **chatbot** es un programa de software diseñado para simular conversaciones humanas a través de interfaces de texto o voz. Los chatbots modernos utilizan técnicas de NLP y machine learning para comprender las intenciones del usuario y proporcionar respuestas relevantes y naturales.

## Tipos de Chatbots

### 1. Chatbots Basados en Reglas

```python
import re
from typing import Dict, List, Optional

class RuleBasedChatbot:
    def __init__(self):
        self.patterns = {
            'saludo': [
                r'\b(hola|buenos días|buenas tardes|buenas noches|hey)\b',
                r'\b(hi|hello|good morning|good afternoon|good evening)\b'
            ],
            'despedida': [
                r'\b(adiós|hasta luego|nos vemos|chao|bye)\b',
                r'\b(goodbye|see you|farewell)\b'
            ],
            'pregunta_nombre': [
                r'\b(cómo te llamas|cuál es tu nombre|quién eres)\b',
                r'\b(what is your name|who are you)\b'
            ],
            'pregunta_ayuda': [
                r'\b(ayuda|help|qué puedes hacer)\b',
                r'\b(what can you do|how can you help)\b'
            ]
        }
        
        self.responses = {
            'saludo': [
                "¡Hola! ¿En qué puedo ayudarte hoy?",
                "¡Buenos días! ¿Cómo estás?",
                "¡Hola! Es un placer saludarte."
            ],
            'despedida': [
                "¡Hasta luego! Que tengas un buen día.",
                "¡Adiós! Espero haberte ayudado.",
                "¡Nos vemos pronto!"
            ],
            'pregunta_nombre': [
                "Soy un chatbot de demostración. Puedes llamarme Bot.",
                "Mi nombre es ChatBot Assistant.",
                "Soy tu asistente virtual."
            ],
            'pregunta_ayuda': [
                "Puedo ayudarte con información básica y responder preguntas simples.",
                "Estoy aquí para asistirte. Puedes preguntarme sobre varios temas.",
                "Puedo responder preguntas, ayudarte con información y mantener una conversación."
            ],
            'default': [
                "No estoy seguro de cómo responder a eso.",
                "¿Podrías reformular tu pregunta?",
                "Interesante. ¿Puedes decirme más?",
                "No tengo una respuesta específica para eso, pero estoy aquí para ayudar."
            ]
        }
    
    def match_pattern(self, user_input: str) -> Optional[str]:
        """Encuentra el patrón que coincide con la entrada del usuario"""
        user_input = user_input.lower()
        
        for intent, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, user_input, re.IGNORECASE):
                    return intent
        return None
    
    def get_response(self, user_input: str) -> str:
        """Genera respuesta basada en patrones"""
        import random
        
        intent = self.match_pattern(user_input)
        
        if intent and intent in self.responses:
            return random.choice(self.responses[intent])
        else:
            return random.choice(self.responses['default'])
    
    def chat(self):
        """Inicia sesión de chat interactiva"""
        print("🤖 Chatbot iniciado. Escribe 'quit' para salir.\n")
        
        while True:
            user_input = input("👤 Tú: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'salir']:
                print("🤖 Bot: ¡Hasta luego!")
                break
            
            if user_input:
                response = self.get_response(user_input)
                print(f"🤖 Bot: {response}\n")

# Ejemplo de uso
bot = RuleBasedChatbot()
# bot.chat()  # Descomenta para probar interactivamente

# Pruebas automáticas
test_inputs = [
    "Hola, ¿cómo estás?",
    "¿Cuál es tu nombre?",
    "¿En qué puedes ayudarme?",
    "¿Qué tiempo hace hoy?",
    "Adiós"
]

print("🧪 Pruebas del chatbot basado en reglas:\n")
for inp in test_inputs:
    response = bot.get_response(inp)
    print(f"👤 Usuario: {inp}")
    print(f"🤖 Bot: {response}\n")
```

### 2. Chatbots con Machine Learning

```python
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle
import json

# Descargar recursos necesarios de NLTK
# nltk.download('punkt')
# nltk.download('stopwords')

class MLChatbot:
    def __init__(self):
        self.pipeline = None
        self.label_encoder = {}
        self.responses = {}
        self.is_trained = False
    
    def prepare_training_data(self, training_file: str = None):
        """Prepara datos de entrenamiento"""
        if training_file:
            with open(training_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            # Datos de ejemplo
            data = {
                "intents": [
                    {
                        "tag": "saludo",
                        "patterns": [
                            "Hola", "Buenos días", "Buenas tardes", "Hey", "Hola, ¿cómo estás?",
                            "Hi", "Hello", "Good morning", "Good afternoon"
                        ],
                        "responses": [
                            "¡Hola! ¿En qué puedo ayudarte?",
                            "¡Buenos días! ¿Cómo puedo asistirte?",
                            "¡Hola! Es un placer saludarte."
                        ]
                    },
                    {
                        "tag": "despedida",
                        "patterns": [
                            "Adiós", "Hasta luego", "Nos vemos", "Chao", "Bye",
                            "Goodbye", "See you later", "Take care"
                        ],
                        "responses": [
                            "¡Hasta luego! Que tengas un buen día.",
                            "¡Adiós! Espero haberte ayudado.",
                            "¡Nos vemos pronto!"
                        ]
                    },
                    {
                        "tag": "informacion_producto",
                        "patterns": [
                            "¿Qué productos tienes?", "Cuéntame sobre tus productos",
                            "¿Qué vendes?", "Información de productos", "Catálogo",
                            "What products do you have?", "Tell me about your products"
                        ],
                        "responses": [
                            "Ofrecemos una amplia gama de productos tecnológicos.",
                            "Tenemos laptops, smartphones, tablets y accesorios.",
                            "Puedes ver nuestro catálogo completo en la página de productos."
                        ]
                    },
                    {
                        "tag": "soporte",
                        "patterns": [
                            "Necesito ayuda", "Tengo un problema", "Soporte técnico",
                            "No funciona", "Error", "Help", "I need support",
                            "Technical support", "I have a problem"
                        ],
                        "responses": [
                            "Estoy aquí para ayudarte. ¿Podrías describir el problema?",
                            "¿Qué tipo de ayuda necesitas?",
                            "Te ayudo con tu consulta. ¿Cuál es el problema específico?"
                        ]
                    },
                    {
                        "tag": "precio",
                        "patterns": [
                            "¿Cuánto cuesta?", "Precio", "¿Qué precio tiene?",
                            "¿Es caro?", "Información de precios", "How much does it cost?",
                            "What's the price?", "Price information"
                        ],
                        "responses": [
                            "Los precios varían según el producto. ¿Qué producto te interesa?",
                            "Puedes consultar precios específicos en nuestro catálogo.",
                            "¿Sobre qué producto quieres conocer el precio?"
                        ]
                    }
                ]
            }
        
        # Extraer patrones y etiquetas
        patterns = []
        labels = []
        
        for intent in data["intents"]:
            tag = intent["tag"]
            self.responses[tag] = intent["responses"]
            
            for pattern in intent["patterns"]:
                patterns.append(pattern.lower())
                labels.append(tag)
        
        return patterns, labels
    
    def train(self, training_file: str = None):
        """Entrena el modelo de clasificación"""
        print("🎓 Entrenando el chatbot...")
        
        patterns, labels = self.prepare_training_data(training_file)
        
        # Crear pipeline de procesamiento
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                lowercase=True,
                stop_words='english',  # Cambiar según idioma
                ngram_range=(1, 2),
                max_features=1000
            )),
            ('classifier', MultinomialNB(alpha=0.1))
        ])
        
        # Entrenar
        self.pipeline.fit(patterns, labels)
        self.is_trained = True
        
        print("✅ Entrenamiento completado!")
    
    def predict_intent(self, user_input: str) -> tuple:
        """Predice la intención del usuario"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")
        
        # Predecir
        intent = self.pipeline.predict([user_input.lower()])[0]
        
        # Obtener probabilidad
        probabilities = self.pipeline.predict_proba([user_input.lower()])[0]
        confidence = max(probabilities)
        
        return intent, confidence
    
    def get_response(self, user_input: str, confidence_threshold: float = 0.3) -> str:
        """Genera respuesta basada en la intención predicha"""
        import random
        
        if not self.is_trained:
            return "Lo siento, el bot aún no está entrenado."
        
        intent, confidence = self.predict_intent(user_input)
        
        if confidence < confidence_threshold:
            return "No estoy seguro de cómo responder a eso. ¿Podrías reformular tu pregunta?"
        
        if intent in self.responses:
            return random.choice(self.responses[intent])
        else:
            return "Lo siento, no tengo una respuesta para eso."
    
    def save_model(self, filepath: str):
        """Guarda el modelo entrenado"""
        model_data = {
            'pipeline': self.pipeline,
            'responses': self.responses,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Modelo guardado en {filepath}")
    
    def load_model(self, filepath: str):
        """Carga un modelo previamente entrenado"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.pipeline = model_data['pipeline']
        self.responses = model_data['responses']
        self.is_trained = model_data['is_trained']
        
        print(f"📁 Modelo cargado desde {filepath}")
    
    def chat(self):
        """Inicia sesión de chat interactiva"""
        if not self.is_trained:
            print("⚠️ El modelo no está entrenado. Entrenando con datos por defecto...")
            self.train()
        
        print("🤖 Chatbot ML iniciado. Escribe 'quit' para salir.\n")
        
        while True:
            user_input = input("👤 Tú: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'salir']:
                print("🤖 Bot: ¡Hasta luego!")
                break
            
            if user_input:
                intent, confidence = self.predict_intent(user_input)
                response = self.get_response(user_input)
                
                print(f"🤖 Bot: {response}")
                print(f"🔍 Intención detectada: {intent} (confianza: {confidence:.2f})\n")

# Ejemplo de uso
ml_bot = MLChatbot()
ml_bot.train()

# Pruebas
test_inputs = [
    "Hola, ¿cómo estás?",
    "¿Qué productos tienes disponibles?",
    "Necesito ayuda con mi pedido",
    "¿Cuánto cuesta el producto?",
    "Adiós, gracias por todo"
]

print("🧪 Pruebas del chatbot con ML:\n")
for inp in test_inputs:
    intent, confidence = ml_bot.predict_intent(inp)
    response = ml_bot.get_response(inp)
    print(f"👤 Usuario: {inp}")
    print(f"🤖 Bot: {response}")
    print(f"🔍 Intención: {intent} (confianza: {confidence:.2f})\n")
```

### 3. Chatbots con Transformers

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

class TransformerChatbot:
    def __init__(self, model_name='microsoft/DialoGPT-medium'):
        print(f"🔄 Cargando modelo {model_name}...")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Configurar token de padding
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.conversation_history = []
        self.max_length = 1000
        
        print("✅ Modelo cargado exitosamente!")
    
    def generate_response(self, user_input: str, max_new_tokens: int = 100) -> str:
        """Genera respuesta usando el modelo transformer"""
        
        # Añadir entrada del usuario al historial
        self.conversation_history.append(user_input)
        
        # Crear contexto de conversación
        conversation = self.tokenizer.eos_token.join(self.conversation_history)
        
        # Tokenizar
        inputs = self.tokenizer.encode(
            conversation, 
            return_tensors='pt',
            max_length=self.max_length,
            truncation=True
        ).to(self.device)
        
        # Generar respuesta
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decodificar respuesta
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extraer solo la nueva respuesta
        if self.tokenizer.eos_token in response:
            response_parts = response.split(self.tokenizer.eos_token)
            new_response = response_parts[-1].strip()
        else:
            new_response = response[len(conversation):].strip()
        
        # Añadir respuesta al historial
        self.conversation_history.append(new_response)
        
        # Mantener historial limitado
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        return new_response
    
    def reset_conversation(self):
        """Reinicia la conversación"""
        self.conversation_history = []
        print("🔄 Conversación reiniciada")
    
    def chat(self):
        """Inicia sesión de chat interactiva"""
        print("🤖 Chatbot Transformer iniciado. Comandos: 'quit' (salir), 'reset' (reiniciar)\n")
        
        while True:
            user_input = input("👤 Tú: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'salir']:
                print("🤖 Bot: ¡Ha sido un placer conversar contigo!")
                break
            elif user_input.lower() == 'reset':
                self.reset_conversation()
                continue
            
            if user_input:
                try:
                    response = self.generate_response(user_input)
                    print(f"🤖 Bot: {response}\n")
                except Exception as e:
                    print(f"❌ Error generando respuesta: {e}\n")

# Versión simplificada usando pipeline
class SimpleChatbot:
    def __init__(self):
        print("🔄 Cargando chatbot simple...")
        self.chatbot = pipeline(
            "conversational",
            model="microsoft/DialoGPT-small",
            device=0 if torch.cuda.is_available() else -1
        )
        print("✅ Chatbot listo!")
    
    def chat_with_pipeline(self, user_input: str, conversation_id: str = "default"):
        """Genera respuesta usando pipeline de conversación"""
        from transformers import Conversation
        
        if not hasattr(self, 'conversations'):
            self.conversations = {}
        
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = Conversation()
        
        # Añadir mensaje del usuario
        self.conversations[conversation_id].add_user_input(user_input)
        
        # Generar respuesta
        result = self.chatbot(self.conversations[conversation_id])
        
        return result.generated_responses[-1]

# Ejemplo usando el chatbot simple
simple_bot = SimpleChatbot()

print("🧪 Pruebas del chatbot simple:\n")
test_conversations = [
    "Hola, ¿cómo estás?",
    "¿Puedes ayudarme con algo?",
    "¿Qué opinas sobre la inteligencia artificial?",
    "Gracias por la conversación"
]

for message in test_conversations:
    response = simple_bot.chat_with_pipeline(message)
    print(f"👤 Usuario: {message}")
    print(f"🤖 Bot: {response}\n")
```

## Chatbots Especializados

### 1. Chatbot de Servicio al Cliente

```python
class CustomerServiceBot:
    def __init__(self):
        self.user_sessions = {}
        self.ticket_counter = 1000
        
        # Base de conocimiento FAQ
        self.faq = {
            "horario": "Nuestro horario de atención es de lunes a viernes de 9:00 AM a 6:00 PM.",
            "devolucion": "Aceptamos devoluciones dentro de 30 días con el recibo de compra.",
            "envio": "El envío estándar toma 3-5 días hábiles. El envío express 1-2 días.",
            "pago": "Aceptamos tarjetas de crédito, débito, PayPal y transferencias bancarias.",
            "garantia": "Todos nuestros productos tienen garantía de 1 año contra defectos de fábrica."
        }
        
        self.ml_bot = MLChatbot()
        self.ml_bot.train()
    
    def get_user_session(self, user_id: str) -> dict:
        """Obtiene o crea sesión de usuario"""
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                'conversation_history': [],
                'current_ticket': None,
                'user_info': {}
            }
        return self.user_sessions[user_id]
    
    def search_faq(self, query: str) -> str:
        """Busca en la base de conocimiento FAQ"""
        query_lower = query.lower()
        
        for keyword, answer in self.faq.items():
            if keyword in query_lower:
                return f"📋 {answer}"
        
        return None
    
    def create_ticket(self, user_id: str, issue: str) -> str:
        """Crea un ticket de soporte"""
        session = self.get_user_session(user_id)
        ticket_id = f"TK-{self.ticket_counter}"
        self.ticket_counter += 1
        
        session['current_ticket'] = {
            'id': ticket_id,
            'issue': issue,
            'status': 'open',
            'created_at': 'now'  # En producción usar datetime
        }
        
        return f"🎫 He creado el ticket {ticket_id} para tu consulta: '{issue}'. Un agente te contactará pronto."
    
    def handle_message(self, user_id: str, message: str) -> str:
        """Maneja mensaje del usuario con lógica de servicio al cliente"""
        session = self.get_user_session(user_id)
        session['conversation_history'].append(('user', message))
        
        # 1. Buscar en FAQ primero
        faq_response = self.search_faq(message)
        if faq_response:
            response = faq_response
        else:
            # 2. Usar ML bot para clasificar intención
            intent, confidence = self.ml_bot.predict_intent(message)
            
            if intent == 'soporte' and confidence > 0.6:
                # Crear ticket para problemas complejos
                response = self.create_ticket(user_id, message)
            else:
                # 3. Respuesta general
                response = self.ml_bot.get_response(message)
        
        session['conversation_history'].append(('bot', response))
        return response
    
    def get_conversation_summary(self, user_id: str) -> dict:
        """Obtiene resumen de la conversación"""
        session = self.get_user_session(user_id)
        return {
            'messages_count': len(session['conversation_history']),
            'has_ticket': session['current_ticket'] is not None,
            'ticket_id': session['current_ticket']['id'] if session['current_ticket'] else None
        }

# Ejemplo de uso
cs_bot = CustomerServiceBot()

# Simulación de conversación de servicio al cliente
conversations = [
    ("user1", "Hola, ¿cuál es su horario de atención?"),
    ("user1", "¿Puedo devolver un producto?"),
    ("user2", "Mi producto llegó dañado y no funciona"),
    ("user2", "¿Cómo puedo obtener un reembolso?"),
    ("user3", "¿Qué formas de pago aceptan?")
]

print("🧪 Simulación de chatbot de servicio al cliente:\n")
for user_id, message in conversations:
    response = cs_bot.handle_message(user_id, message)
    print(f"👤 {user_id}: {message}")
    print(f"🤖 Bot: {response}")
    
    # Mostrar resumen de sesión
    summary = cs_bot.get_conversation_summary(user_id)
    if summary['has_ticket']:
        print(f"🎫 Ticket creado: {summary['ticket_id']}")
    print()
```

### 2. Chatbot Educativo

```python
class EducationalChatbot:
    def __init__(self):
        self.topics = {
            'matematicas': {
                'subtopics': ['algebra', 'geometria', 'calculo'],
                'examples': {
                    'algebra': "Ejemplo: Resolver 2x + 5 = 15\nSolución: 2x = 10, x = 5",
                    'geometria': "Ejemplo: Área de un círculo = π × r²",
                    'calculo': "Ejemplo: Derivada de x² = 2x"
                }
            },
            'programacion': {
                'subtopics': ['python', 'javascript', 'algoritmos'],
                'examples': {
                    'python': "print('Hola, mundo!')",
                    'javascript': "console.log('Hola, mundo!');",
                    'algoritmos': "Algoritmo de búsqueda binaria divide el array por la mitad"
                }
            }
        }
        
        self.student_progress = {}
    
    def track_progress(self, student_id: str, topic: str, subtopic: str):
        """Rastrea progreso del estudiante"""
        if student_id not in self.student_progress:
            self.student_progress[student_id] = {}
        
        if topic not in self.student_progress[student_id]:
            self.student_progress[student_id][topic] = set()
        
        self.student_progress[student_id][topic].add(subtopic)
    
    def get_recommendation(self, student_id: str) -> str:
        """Recomienda siguiente tema basado en progreso"""
        if student_id not in self.student_progress:
            return "Te recomiendo comenzar con matemáticas básicas o programación en Python."
        
        progress = self.student_progress[student_id]
        
        for topic, subtopics in self.topics.items():
            completed = progress.get(topic, set())
            remaining = set(subtopics['subtopics']) - completed
            
            if remaining:
                next_topic = list(remaining)[0]
                return f"Basado en tu progreso, te recomiendo continuar con {next_topic} en {topic}."
        
        return "¡Excelente! Has completado todos los temas disponibles."
    
    def explain_topic(self, topic: str, subtopic: str = None) -> str:
        """Explica un tema específico"""
        if topic not in self.topics:
            return f"Lo siento, no tengo información sobre {topic}."
        
        if subtopic is None:
            subtopics = ', '.join(self.topics[topic]['subtopics'])
            return f"En {topic} puedo ayudarte con: {subtopics}. ¿Sobre cuál te gustaría aprender?"
        
        if subtopic in self.topics[topic]['examples']:
            example = self.topics[topic]['examples'][subtopic]
            return f"Aquí tienes información sobre {subtopic}:\n\n{example}"
        
        return f"No tengo ejemplos específicos para {subtopic} en {topic}."
    
    def generate_quiz(self, topic: str) -> dict:
        """Genera quiz simple para un tema"""
        quizzes = {
            'matematicas': {
                'question': "¿Cuál es la derivada de x³?",
                'options': ['3x²', '2x²', 'x²', '3x'],
                'answer': '3x²'
            },
            'programacion': {
                'question': "¿Qué imprime print(2 + 3 * 4)?",
                'options': ['20', '14', '11', '9'],
                'answer': '14'
            }
        }
        
        return quizzes.get(topic, {
            'question': 'Quiz no disponible para este tema',
            'options': [],
            'answer': ''
        })

# Ejemplo de uso del chatbot educativo
edu_bot = EducationalChatbot()

print("🧪 Simulación de chatbot educativo:\n")

# Estudiante pregunta sobre matemáticas
response = edu_bot.explain_topic('matematicas')
print(f"🤖 Bot: {response}\n")

# Estudiante específica álgebra
response = edu_bot.explain_topic('matematicas', 'algebra')
print(f"🤖 Bot: {response}\n")

# Rastrear progreso
edu_bot.track_progress('estudiante1', 'matematicas', 'algebra')

# Obtener recomendación
recommendation = edu_bot.get_recommendation('estudiante1')
print(f"🤖 Recomendación: {recommendation}\n")

# Generar quiz
quiz = edu_bot.generate_quiz('matematicas')
print(f"🤖 Quiz: {quiz['question']}")
print(f"   Opciones: {', '.join(quiz['options'])}")
print(f"   Respuesta correcta: {quiz['answer']}\n")
```

## Integración con Plataformas

### 1. Chatbot para Discord

```python
import discord
from discord.ext import commands

class DiscordChatbot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix='!', intents=intents)
        
        self.chatbot = MLChatbot()
        self.chatbot.train()
    
    async def on_ready(self):
        print(f'🤖 {self.user} está conectado a Discord!')
    
    async def on_message(self, message):
        # Ignorar mensajes del bot
        if message.author == self.user:
            return
        
        # Responder a mensajes directos o menciones
        if isinstance(message.channel, discord.DMChannel) or self.user.mentioned_in(message):
            user_message = message.content.replace(f'<@{self.user.id}>', '').strip()
            
            if user_message:
                try:
                    response = self.chatbot.get_response(user_message)
                    await message.channel.send(response)
                except Exception as e:
                    await message.channel.send("Lo siento, hubo un error procesando tu mensaje.")
        
        # Procesar comandos
        await self.process_commands(message)
    
    @commands.command(name='chat')
    async def chat_command(self, ctx, *, message):
        """Comando para chatear con el bot"""
        response = self.chatbot.get_response(message)
        await ctx.send(response)
    
    @commands.command(name='ayuda_bot')
    async def help_bot(self, ctx):
        """Ayuda del chatbot"""
        help_text = """
        🤖 **Comandos del Chatbot:**
        
        `!chat <mensaje>` - Chatear con el bot
        `!ayuda_bot` - Mostrar esta ayuda
        
        También puedes mencionarme (@bot) o enviarme un mensaje directo.
        """
        await ctx.send(help_text)

# Para usar: bot = DiscordChatbot()
# bot.run('TU_TOKEN_DE_DISCORD')
```

### 2. Chatbot Web con Flask

```python
from flask import Flask, request, jsonify, render_template_string
import uuid

app = Flask(__name__)

# HTML template simple para la interfaz web
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Chatbot Web</title>
    <style>
        .chat-container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .chat-messages { height: 400px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px; margin-bottom: 10px; }
        .message { margin: 10px 0; }
        .user-message { text-align: right; color: blue; }
        .bot-message { text-align: left; color: green; }
        .input-area { display: flex; }
        .input-area input { flex: 1; padding: 10px; }
        .input-area button { padding: 10px 20px; }
    </style>
</head>
<body>
    <div class="chat-container">
        <h1>🤖 Chatbot Web</h1>
        <div id="chat-messages" class="chat-messages"></div>
        <div class="input-area">
            <input type="text" id="user-input" placeholder="Escribe tu mensaje...">
            <button onclick="sendMessage()">Enviar</button>
        </div>
    </div>

    <script>
        let sessionId = Math.random().toString(36).substring(7);
        
        function addMessage(message, isUser) {
            const messagesDiv = document.getElementById('chat-messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message ' + (isUser ? 'user-message' : 'bot-message');
            messageDiv.textContent = (isUser ? 'Tú: ' : 'Bot: ') + message;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        async function sendMessage() {
            const input = document.getElementById('user-input');
            const message = input.value.trim();
            
            if (!message) return;
            
            addMessage(message, true);
            input.value = '';
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: message, session_id: sessionId })
                });
                
                const data = await response.json();
                addMessage(data.response, false);
            } catch (error) {
                addMessage('Error: No se pudo conectar con el bot', false);
            }
        }
        
        // Enviar mensaje con Enter
        document.getElementById('user-input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
        
        // Mensaje de bienvenida
        window.onload = function() {
            addMessage('¡Hola! Soy tu chatbot asistente. ¿En qué puedo ayudarte?', false);
        };
    </script>
</body>
</html>
"""

class WebChatbot:
    def __init__(self):
        self.chatbot = MLChatbot()
        self.chatbot.train()
        self.sessions = {}
    
    def get_session(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'conversation_history': []
            }
        return self.sessions[session_id]
    
    def handle_message(self, message, session_id):
        session = self.get_session(session_id)
        session['conversation_history'].append(('user', message))
        
        # Generar respuesta
        response = self.chatbot.get_response(message)
        session['conversation_history'].append(('bot', response))
        
        return response

web_chatbot = WebChatbot()

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '')
    session_id = data.get('session_id', str(uuid.uuid4()))
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        response = web_chatbot.handle_message(message, session_id)
        return jsonify({
            'response': response,
            'session_id': session_id
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    print("🌐 Iniciando servidor web del chatbot...")
    print("📱 Visita http://localhost:5000 para usar el chatbot")
    app.run(debug=True, port=5000)
```

## Mejores Prácticas

### 1. Manejo de Contexto

```python
class ContextAwareChatbot:
    def __init__(self):
        self.context_window = 5  # Mantener últimos 5 intercambios
        self.user_contexts = {}
    
    def update_context(self, user_id: str, user_message: str, bot_response: str):
        """Actualiza el contexto conversacional"""
        if user_id not in self.user_contexts:
            self.user_contexts[user_id] = []
        
        self.user_contexts[user_id].append({
            'user': user_message,
            'bot': bot_response,
            'timestamp': 'now'  # En producción usar datetime
        })
        
        # Mantener solo los últimos intercambios
        if len(self.user_contexts[user_id]) > self.context_window:
            self.user_contexts[user_id] = self.user_contexts[user_id][-self.context_window:]
    
    def get_contextual_response(self, user_id: str, message: str) -> str:
        """Genera respuesta considerando el contexto"""
        context = self.user_contexts.get(user_id, [])
        
        # Construir prompt con contexto
        context_text = ""
        for exchange in context[-3:]:  # Últimos 3 intercambios
            context_text += f"Usuario: {exchange['user']}\nBot: {exchange['bot']}\n"
        
        # Aquí integrarías con tu modelo de elección
        # Por ahora, respuesta simple basada en contexto
        contextual_message = f"Contexto: {context_text}\nUsuario actual: {message}"
        
        # Simular respuesta contextual
        if context and "gracias" in message.lower():
            return "¡De nada! ¿Hay algo más en lo que pueda ayudarte?"
        
        return f"Entiendo tu mensaje: '{message}'"
```

### 2. Evaluación y Métricas

```python
class ChatbotEvaluator:
    def __init__(self):
        self.conversation_logs = []
        self.user_ratings = []
    
    def log_conversation(self, user_id: str, conversation: list):
        """Registra conversación para análisis"""
        self.conversation_logs.append({
            'user_id': user_id,
            'conversation': conversation,
            'timestamp': 'now'
        })
    
    def collect_user_rating(self, user_id: str, rating: int, feedback: str = ""):
        """Recolecta calificación del usuario"""
        self.user_ratings.append({
            'user_id': user_id,
            'rating': rating,  # 1-5
            'feedback': feedback,
            'timestamp': 'now'
        })
    
    def calculate_metrics(self) -> dict:
        """Calcula métricas del chatbot"""
        if not self.user_ratings:
            return {'average_rating': 0, 'total_ratings': 0}
        
        ratings = [r['rating'] for r in self.user_ratings]
        
        return {
            'average_rating': sum(ratings) / len(ratings),
            'total_ratings': len(ratings),
            'rating_distribution': {
                i: ratings.count(i) for i in range(1, 6)
            }
        }
    
    def analyze_common_issues(self) -> list:
        """Analiza problemas comunes basados en feedback"""
        negative_feedback = [
            r['feedback'] for r in self.user_ratings 
            if r['rating'] <= 2 and r['feedback']
        ]
        
        # Análisis simple de palabras clave en feedback negativo
        common_issues = []
        keywords = ['lento', 'no entiende', 'error', 'confuso', 'malo']
        
        for keyword in keywords:
            count = sum(1 for feedback in negative_feedback if keyword in feedback.lower())
            if count > 0:
                common_issues.append({'issue': keyword, 'count': count})
        
        return sorted(common_issues, key=lambda x: x['count'], reverse=True)
```

### 3. Seguridad y Moderación

```python
import re

class ChatbotModerator:
    def __init__(self):
        self.blocked_words = [
            # Añadir palabras inapropiadas según contexto
            'spam', 'inappropriate', 'offensive'
        ]
        
        self.suspicious_patterns = [
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Números de tarjeta
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Emails
        ]
    
    def is_safe_message(self, message: str) -> tuple:
        """Verifica si el mensaje es seguro"""
        message_lower = message.lower()
        
        # Verificar palabras bloqueadas
        for word in self.blocked_words:
            if word in message_lower:
                return False, f"Mensaje contiene contenido inapropiado: {word}"
        
        # Verificar patrones sospechosos
        for pattern in self.suspicious_patterns:
            if re.search(pattern, message):
                return False, "Mensaje contiene información sensible detectada"
        
        return True, "Mensaje aprobado"
    
    def moderate_response(self, response: str) -> str:
        """Modera la respuesta del bot"""
        # Evitar compartir información sensible
        if any(word in response.lower() for word in ['password', 'contraseña', 'pin']):
            return "Lo siento, no puedo ayudar con información de seguridad."
        
        return response

# Integración con chatbot moderado
class ModeratedChatbot(MLChatbot):
    def __init__(self):
        super().__init__()
        self.moderator = ChatbotModerator()
    
    def safe_response(self, user_input: str) -> dict:
        """Genera respuesta con moderación"""
        # Verificar entrada
        is_safe, safety_message = self.moderator.is_safe_message(user_input)
        
        if not is_safe:
            return {
                'response': "Lo siento, no puedo procesar ese mensaje.",
                'safe': False,
                'reason': safety_message
            }
        
        # Generar respuesta normal
        response = self.get_response(user_input)
        
        # Moderar respuesta
        moderated_response = self.moderator.moderate_response(response)
        
        return {
            'response': moderated_response,
            'safe': True,
            'reason': None
        }
```

Los chatbots representan una de las aplicaciones más visibles y útiles del NLP, ofreciendo interfaces naturales para interactuar con sistemas digitales. La clave del éxito está en elegir la arquitectura adecuada para cada caso de uso y implementar mejores prácticas de seguridad, evaluación y experiencia de usuario.

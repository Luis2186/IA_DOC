---
layout: ../../layouts/DocLayout.astro
title: "Large Language Models (LLMs)"
description: "Modelos de lenguaje grandes y su impacto en la IA"
currentPath: "/deep-learning/llms"
---

# Large Language Models (LLMs)

Los Large Language Models representan una revolución en el procesamiento de lenguaje natural, demostrando capacidades emergentes que van desde la comprensión hasta la generación creativa de texto.

## ¿Qué son los LLMs?

### Definición
Los LLMs son modelos de redes neuronales masivos, típicamente basados en arquitectura Transformer, entrenados en enormes cantidades de texto para predecir la siguiente palabra en una secuencia.

### Características Clave
- **Escala masiva**: Millones a billones de parámetros
- **Entrenamiento no supervisado**: Aprenden patrones del lenguaje
- **Capacidades emergentes**: Habilidades que surgen con el tamaño
- **Few-shot learning**: Aprenden nuevas tareas con pocos ejemplos

## Evolución de los LLMs

### Timeline Histórico

| Año | Modelo | Parámetros | Organización | Innovación Clave |
|-----|--------|------------|--------------|------------------|
| 2018 | GPT-1 | 117M | OpenAI | Transformer decoder |
| 2019 | GPT-2 | 1.5B | OpenAI | Scaling up |
| 2020 | GPT-3 | 175B | OpenAI | Few-shot learning |
| 2021 | PaLM | 540B | Google | Breakthrough scale |
| 2022 | ChatGPT | ~175B | OpenAI | RLHF |
| 2023 | GPT-4 | ~1.7T | OpenAI | Multimodal |
| 2024 | Claude-3 | ~? | Anthropic | Constitutional AI |

## Arquitectura y Entrenamiento

### 1. Arquitectura Base
```python
import torch
import torch.nn as nn
import math

class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_len):
        super(GPTModel, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        
        # Output layer
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying
        self.head.weight = self.token_embedding.weight
        
    def forward(self, input_ids, position_ids=None):
        batch_size, seq_len = input_ids.shape
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device)
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(position_ids)
        x = token_emb + pos_emb
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Output projection
        logits = self.head(x)
        
        return logits

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.mlp = MLP(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # Pre-norm architecture
        x = x + self.attention(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
```

### 2. Proceso de Entrenamiento

#### Pre-entrenamiento
```python
def train_llm(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        
        # Forward pass
        logits = model(input_ids[:, :-1])  # Predecir siguiente token
        targets = input_ids[:, 1:]  # Targets shifted
        
        # Calcular loss
        loss = nn.CrossEntropyLoss()(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    return total_loss / len(dataloader)
```

#### Instruction Tuning
```python
def instruction_tuning_step(model, batch, optimizer):
    """Fine-tuning en formato instrucción-respuesta"""
    
    instructions = batch['instructions']
    responses = batch['responses']
    
    # Formato: "Instrucción: {instrucción}\nRespuesta: {respuesta}"
    inputs = [f"Instrucción: {inst}\nRespuesta: {resp}" 
              for inst, resp in zip(instructions, responses)]
    
    # Tokenizar
    tokenized = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True)
    
    # Forward pass
    outputs = model(tokenized['input_ids'])
    
    # Calcular loss solo en la parte de respuesta
    loss = compute_response_loss(outputs, tokenized, responses)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

### 3. RLHF (Reinforcement Learning from Human Feedback)

```python
class RLHFTrainer:
    def __init__(self, policy_model, value_model, reference_model):
        self.policy_model = policy_model
        self.value_model = value_model
        self.reference_model = reference_model
        
    def compute_ppo_loss(self, responses, rewards, old_log_probs):
        """Implementar PPO loss para RLHF"""
        
        # Nuevas probabilidades
        new_log_probs = self.policy_model.get_log_probs(responses)
        
        # Ratio de probabilidades
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # KL penalty con modelo de referencia
        ref_log_probs = self.reference_model.get_log_probs(responses)
        kl_penalty = new_log_probs - ref_log_probs
        
        # Ajustar rewards
        adjusted_rewards = rewards - 0.1 * kl_penalty
        
        # Ventajas
        values = self.value_model(responses)
        advantages = adjusted_rewards - values
        
        # PPO clipped loss
        clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
        loss1 = ratio * advantages
        loss2 = clipped_ratio * advantages
        
        policy_loss = -torch.min(loss1, loss2).mean()
        value_loss = nn.MSELoss()(values, adjusted_rewards)
        
        return policy_loss + 0.5 * value_loss
    
    def train_step(self, prompts, responses, rewards):
        # Generar respuestas con modelo actual
        with torch.no_grad():
            old_log_probs = self.policy_model.get_log_probs(responses)
        
        # Computar loss
        loss = self.compute_ppo_loss(responses, rewards, old_log_probs)
        
        # Actualizar
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

## Capacidades Emergentes

### 1. Few-Shot Learning
```python
def few_shot_prompting(model, task_examples, new_input):
    """Realizar few-shot learning con ejemplos en contexto"""
    
    # Construir prompt con ejemplos
    prompt = "Realiza la siguiente tarea basándote en estos ejemplos:\n\n"
    
    for i, example in enumerate(task_examples):
        prompt += f"Ejemplo {i+1}:\n"
        prompt += f"Entrada: {example['input']}\n"
        prompt += f"Salida: {example['output']}\n\n"
    
    prompt += f"Ahora resuelve:\nEntrada: {new_input}\nSalida:"
    
    # Generar respuesta
    response = model.generate(prompt, max_length=100, temperature=0.3)
    
    return response

# Ejemplo de uso
examples = [
    {"input": "2 + 3", "output": "5"},
    {"input": "7 - 4", "output": "3"},
    {"input": "5 * 2", "output": "10"}
]

result = few_shot_prompting(model, examples, "8 + 6")
```

### 2. Chain-of-Thought Reasoning
```python
def chain_of_thought_prompting(model, problem):
    """Prompting con razonamiento paso a paso"""
    
    cot_prompt = f"""
Resuelve este problema paso a paso:

Problema: {problem}

Pensemos paso a paso:
1. Primero, identifiquemos qué información tenemos
2. Luego, determinemos qué necesitamos calcular
3. Finalmente, resolvamos paso a paso

Solución:
"""
    
    response = model.generate(
        cot_prompt, 
        max_length=300, 
        temperature=0.1,
        do_sample=True
    )
    
    return response

# Ejemplo
problem = "Un tren viaja a 60 km/h durante 2 horas, luego a 80 km/h durante 1.5 horas. ¿Cuál es la distancia total recorrida?"
solution = chain_of_thought_prompting(model, problem)
```

### 3. Tool Use y Function Calling
```python
class ToolUsingLLM:
    def __init__(self, base_model, available_tools):
        self.model = base_model
        self.tools = available_tools
        
    def parse_tool_call(self, response):
        """Extraer llamadas a herramientas del texto"""
        import re
        
        # Buscar patrones como: TOOL_CALL(tool_name, arg1, arg2)
        pattern = r'TOOL_CALL\((\w+),\s*([^)]+)\)'
        matches = re.findall(pattern, response)
        
        tool_calls = []
        for tool_name, args_str in matches:
            if tool_name in self.tools:
                args = [arg.strip().strip('"\'') for arg in args_str.split(',')]
                tool_calls.append((tool_name, args))
        
        return tool_calls
    
    def execute_tools(self, tool_calls):
        """Ejecutar las herramientas llamadas"""
        results = {}
        
        for tool_name, args in tool_calls:
            try:
                result = self.tools[tool_name](*args)
                results[tool_name] = result
            except Exception as e:
                results[tool_name] = f"Error: {str(e)}"
        
        return results
    
    def generate_with_tools(self, prompt):
        """Generar respuesta con acceso a herramientas"""
        
        # Agregar información sobre herramientas disponibles
        tool_info = "Herramientas disponibles:\n"
        for tool_name, tool_func in self.tools.items():
            tool_info += f"- {tool_name}: {tool_func.__doc__}\n"
        
        enhanced_prompt = f"{tool_info}\n{prompt}\n\nPuedes usar herramientas con: TOOL_CALL(tool_name, arg1, arg2)"
        
        # Generar respuesta inicial
        response = self.model.generate(enhanced_prompt)
        
        # Buscar y ejecutar herramientas
        tool_calls = self.parse_tool_call(response)
        
        if tool_calls:
            tool_results = self.execute_tools(tool_calls)
            
            # Generar respuesta final con resultados
            final_prompt = f"{enhanced_prompt}\n\nRespuesta inicial: {response}\n\nResultados de herramientas: {tool_results}\n\nRespuesta final:"
            final_response = self.model.generate(final_prompt)
            
            return final_response
        
        return response

# Definir herramientas
def calculate(expression):
    """Evalúa expresiones matemáticas"""
    try:
        return eval(expression)
    except:
        return "Error en cálculo"

def search_web(query):
    """Busca información en la web"""
    # Implementación simplificada
    return f"Resultados para: {query}"

tools = {
    'calculate': calculate,
    'search_web': search_web
}

# Usar LLM con herramientas
tool_llm = ToolUsingLLM(model, tools)
response = tool_llm.generate_with_tools("¿Cuánto es 15% de 240?")
```

## Técnicas de Optimización

### 1. Efficient Attention
```python
class FlashAttention(nn.Module):
    """Implementación simplificada de Flash Attention"""
    
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        
    def forward(self, x, block_size=512):
        B, N, D = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        
        # Procesar en bloques para eficiencia de memoria
        output = torch.zeros_like(v)
        
        for i in range(0, N, block_size):
            end_i = min(i + block_size, N)
            q_block = q[:, :, i:end_i]
            
            for j in range(0, N, block_size):
                end_j = min(j + block_size, N)
                k_block = k[:, :, j:end_j]
                v_block = v[:, :, j:end_j]
                
                # Atención local
                scores = torch.matmul(q_block, k_block.transpose(-2, -1)) / math.sqrt(self.head_dim)
                attn = torch.softmax(scores, dim=-1)
                output[:, :, i:end_i] += torch.matmul(attn, v_block)
        
        output = output.transpose(1, 2).reshape(B, N, D)
        return self.out(output)
```

### 2. Model Parallelism
```python
class DistributedLLM(nn.Module):
    """LLM distribuido en múltiples GPUs"""
    
    def __init__(self, config, device_map):
        super().__init__()
        self.device_map = device_map
        
        # Distribuir capas en dispositivos
        self.embedding = nn.Embedding(config.vocab_size, config.d_model).to(device_map['embedding'])
        
        self.layers = nn.ModuleList()
        for i in range(config.n_layers):
            device = device_map[f'layer_{i}']
            self.layers.append(TransformerBlock(config.d_model, config.n_heads).to(device))
        
        self.ln_f = nn.LayerNorm(config.d_model).to(device_map['output'])
        self.head = nn.Linear(config.d_model, config.vocab_size).to(device_map['output'])
    
    def forward(self, input_ids):
        x = self.embedding(input_ids.to(self.device_map['embedding']))
        
        for i, layer in enumerate(self.layers):
            layer_device = self.device_map[f'layer_{i}']
            x = layer(x.to(layer_device))
        
        x = x.to(self.device_map['output'])
        x = self.ln_f(x)
        return self.head(x)
```

### 3. Quantization
```python
def quantize_model(model, bits=8):
    """Cuantizar modelo para reducir memoria"""
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Cuantización de pesos
            weight = module.weight.data
            
            # Calcular escala y zero point
            min_val = weight.min()
            max_val = weight.max()
            scale = (max_val - min_val) / (2**bits - 1)
            zero_point = -min_val / scale
            
            # Cuantizar
            quantized_weight = torch.round(weight / scale + zero_point)
            quantized_weight = torch.clamp(quantized_weight, 0, 2**bits - 1)
            
            # Dequantizar para uso
            dequantized_weight = (quantized_weight - zero_point) * scale
            
            module.weight.data = dequantized_weight
    
    return model
```

## Evaluación de LLMs

### 1. Benchmarks Estándar
```python
def evaluate_llm_benchmarks(model, tokenizer):
    """Evaluar modelo en benchmarks estándar"""
    
    benchmarks = {
        'MMLU': evaluate_mmlu,
        'HellaSwag': evaluate_hellaswag,
        'ARC': evaluate_arc,
        'TruthfulQA': evaluate_truthfulqa
    }
    
    results = {}
    
    for benchmark_name, eval_func in benchmarks.items():
        print(f"Evaluando {benchmark_name}...")
        score = eval_func(model, tokenizer)
        results[benchmark_name] = score
        print(f"{benchmark_name}: {score:.2f}")
    
    return results

def evaluate_mmlu(model, tokenizer):
    """Evaluar en MMLU (Massive Multitask Language Understanding)"""
    
    # Cargar preguntas MMLU
    questions = load_mmlu_questions()
    correct = 0
    total = 0
    
    for question in questions:
        prompt = f"Pregunta: {question['question']}\n"
        for i, choice in enumerate(question['choices']):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += "Respuesta:"
        
        # Generar respuesta
        response = model.generate(prompt, max_length=1, temperature=0)
        
        # Extraer letra de respuesta
        predicted = response.strip().upper()
        if predicted in ['A', 'B', 'C', 'D']:
            if predicted == question['answer']:
                correct += 1
        
        total += 1
    
    return correct / total
```

### 2. Evaluación de Sesgo
```python
def evaluate_bias(model, tokenizer):
    """Evaluar sesgos en el modelo"""
    
    bias_tests = {
        'gender': [
            "The nurse prepared her/his equipment",
            "The engineer reviewed her/his designs",
            "The teacher graded her/his papers"
        ],
        'race': [
            "The person from [GROUP] was very intelligent",
            "The [GROUP] student excelled in mathematics"
        ],
        'religion': [
            "People who practice [RELIGION] are generally",
            "The [RELIGION] community is known for"
        ]
    }
    
    bias_scores = {}
    
    for bias_type, templates in bias_tests.items():
        scores = []
        
        for template in templates:
            # Evaluar con diferentes grupos
            group_scores = test_template_bias(model, tokenizer, template, bias_type)
            scores.extend(group_scores)
        
        bias_scores[bias_type] = {
            'mean_difference': np.mean(scores),
            'max_difference': np.max(scores),
            'std_difference': np.std(scores)
        }
    
    return bias_scores
```

### 3. Métricas de Calidad
```python
def evaluate_generation_quality(model, prompts, reference_texts):
    """Evaluar calidad de generación"""
    
    from rouge_score import rouge_scorer
    from sacrebleu import sentence_bleu
    
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    
    rouge_scores = []
    bleu_scores = []
    
    for prompt, reference in zip(prompts, reference_texts):
        # Generar texto
        generated = model.generate(prompt, max_length=200, temperature=0.7)
        
        # ROUGE score
        rouge_score = rouge.score(reference, generated)
        rouge_scores.append(rouge_score)
        
        # BLEU score
        bleu_score = sentence_bleu([reference.split()], generated.split())
        bleu_scores.append(bleu_score)
    
    return {
        'rouge1': np.mean([score['rouge1'].fmeasure for score in rouge_scores]),
        'rouge2': np.mean([score['rouge2'].fmeasure for score in rouge_scores]),
        'rougeL': np.mean([score['rougeL'].fmeasure for score in rouge_scores]),
        'bleu': np.mean(bleu_scores)
    }
```

## Consideraciones Éticas y de Seguridad

### 1. Alineación de IA
```python
class ConstitutionalAI:
    """Implementación simplificada de Constitutional AI"""
    
    def __init__(self, model, constitution):
        self.model = model
        self.constitution = constitution
    
    def critique_response(self, prompt, response):
        """Criticar respuesta basándose en la constitución"""
        
        critique_prompt = f"""
Evalúa la siguiente respuesta basándote en estos principios:
{self.constitution}

Prompt: {prompt}
Respuesta: {response}

¿La respuesta viola algún principio? ¿Cómo se puede mejorar?
Crítica:
"""
        
        critique = self.model.generate(critique_prompt)
        return critique
    
    def revise_response(self, prompt, response, critique):
        """Revisar respuesta basándose en la crítica"""
        
        revision_prompt = f"""
Prompt original: {prompt}
Respuesta original: {response}
Crítica: {critique}

Por favor, revisa la respuesta original para abordar los puntos mencionados en la crítica:
Respuesta revisada:
"""
        
        revised_response = self.model.generate(revision_prompt)
        return revised_response
    
    def generate_safely(self, prompt, iterations=3):
        """Generar respuesta con múltiples rondas de revisión"""
        
        response = self.model.generate(prompt)
        
        for i in range(iterations):
            critique = self.critique_response(prompt, response)
            
            if "no viola" in critique.lower():
                break
                
            response = self.revise_response(prompt, response, critique)
        
        return response
```

### 2. Detección de Contenido Generado
```python
class AITextDetector:
    """Detector de texto generado por IA"""
    
    def __init__(self, model):
        self.model = model
        
    def compute_perplexity(self, text):
        """Calcular perplejidad del texto"""
        
        tokens = self.tokenizer.encode(text)
        total_log_prob = 0
        total_tokens = 0
        
        for i in range(1, len(tokens)):
            context = tokens[:i]
            target = tokens[i]
            
            logits = self.model(torch.tensor([context]))
            log_probs = torch.log_softmax(logits[0, -1], dim=-1)
            
            total_log_prob += log_probs[target].item()
            total_tokens += 1
        
        perplexity = math.exp(-total_log_prob / total_tokens)
        return perplexity
    
    def detect_ai_text(self, text, threshold=50):
        """Detectar si el texto fue generado por IA"""
        
        perplexity = self.compute_perplexity(text)
        
        # Texto con perplejidad muy baja podría ser generado por IA
        if perplexity < threshold:
            return True, perplexity
        else:
            return False, perplexity
```

## Futuro de los LLMs

### Tendencias Emergentes
1. **Modelos Multimodales**: Integración de texto, imagen, audio y video
2. **Agentes Autónomos**: LLMs que pueden planificar y ejecutar tareas complejas
3. **Especialización**: Modelos específicos para dominios particulares
4. **Eficiencia**: Modelos más pequeños pero igual de capaces
5. **Personalización**: Adaptación a usuarios y contextos específicos

### Desafíos Actuales
- **Alucinaciones**: Generación de información falsa
- **Sesgo**: Perpetuación de prejuicios sociales
- **Consumo energético**: Impacto ambiental del entrenamiento
- **Derechos de autor**: Uso de datos protegidos
- **Desplazamiento laboral**: Impacto en el empleo

Los LLMs continúan evolucionando rápidamente, prometiendo transformar múltiples aspectos de la sociedad mientras plantean importantes desafíos técnicos y éticos que la comunidad debe abordar responsablemente.

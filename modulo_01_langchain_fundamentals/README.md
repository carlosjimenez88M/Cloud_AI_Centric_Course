# Modulo 01: Fundamentos de LangChain y LangGraph

## Objetivo del modulo

Dominar los fundamentos de LangChain como framework para construir aplicaciones impulsadas por LLMs. Al finalizar este modulo tendras las bases para construir chatbots, sistemas RAG, herramientas de resumen y agentes autonomos.

---

## Leccion 01: Introduccion a LangChain

### Que es LangChain?

LangChain es un **framework para construir aplicaciones impulsadas por large language models (LLMs)**. Simplifica la integracion de LLMs, facilitando el desarrollo de soluciones como chatbots, sistemas de retrieval-augmented generation (RAG), herramientas de resumen de documentos y agentes autonomos.

### Por que usar LangChain?

LangChain proporciona **abstracciones** que conectan LLMs con diversas herramientas:

- **Servicios de almacenamiento en la nube** para gestionar datos.
- **Herramientas de web scraping** para obtener informacion en tiempo real.
- **Vector databases** para busqueda y recuperacion eficiente.

Esto permite centrarse en **construir soluciones de IA** en lugar de manejar integraciones complejas.

### Componentes principales

#### Models

- Soporta **multiples proveedores de LLM**: OpenAI, Anthropic, Google, etc.
- Usa paquetes de integracion ligeros como `langchain-openai` o `langchain-anthropic`.
- Simplifica la interaccion mediante el metodo `invoke()`.

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-5-mini", temperature=None)
llm.invoke("Cual es la capital de Colombia?")
```

#### Messages

LangChain estandariza la comunicacion con modelos a traves de objetos de mensaje:

| Tipo | Descripcion |
|------|-------------|
| `HumanMessage` | Entrada del usuario |
| `AIMessage` | Respuesta del modelo |
| `SystemMessage` | Instrucciones de contexto |
| `ToolMessage` | Para function calling |

```python
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

messages = [
    SystemMessage("You are a geography tutor"),
    HumanMessage("What's the capital of Brazil?"),
]
llm.invoke(messages)
```

### Interacciones con estado

Los LLMs son **stateless** — no recuerdan interacciones previas. Para conversaciones coherentes, la aplicacion debe gestionar el **chat history** y proporcionar contexto con cada solicitud.

Una conversacion se estructura como una **lista de mensajes** que se pasa al modelo:

```python
messages = [
    SystemMessage("You are a geography tutor"),
    HumanMessage("What's the capital of Brazil?"),
    AIMessage("The capital of Brazil is Brasilia."),
    HumanMessage("And what about Argentina?"),
]
llm.invoke(messages)
```

### Few-Shot Prompting

Al estructurar programaticamente el chat history, se pueden crear **ejemplos de interacciones ideales** que guian al modelo hacia mejores respuestas.

**Trade-offs:**
- Mas ejemplos mejoran la calidad de las respuestas
- Prompts mas grandes aumentan costos y latencia

### Prompt Templates

```python
# Template basico
prompt_template = PromptTemplate(template="Tell me a joke about {topic}")
llm.invoke(prompt_template.format(topic="Java"))

# ChatPromptTemplate para conversaciones
template = ChatPromptTemplate([
    ("system", "You are a helpful AI bot. Your name is {name}."),
    ("human", "{user_input}"),
])

# FewShotPromptTemplate con ejemplos
template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input"],
)
```

> **Archivos:** [`01_introduccion_langchain/demo.py`](01_introduccion_langchain/demo.py) | [`01_introduccion_langchain/exercise.py`](01_introduccion_langchain/exercise.py)

---

## Leccion 02: Structured Outputs

### Por que outputs estructurados?

Los agentes son mas poderosos cuando retornan **outputs estructurados y legibles por maquina** (JSON) en lugar de solo texto plano. Esto permite integracion con otros sistemas, automatizacion de procesos y activacion de acciones en workflows.

```json
{
  "issue_type": "login_problem",
  "urgency": "high",
  "customer_email": "jane@mail.com"
}
```

El texto libre puede ser legible, pero **los datos estructurados son los que permiten a las herramientas actuar**.

### Por que el prompting no es suficiente

Pedirle al modelo que genere JSON mediante prompts a veces funciona, pero los modelos frecuentemente retornan datos vagos o invalidos. Los LLMs estan entrenados para lenguaje natural, no para reglas estrictas de tipos.

### Modelando datos con Pydantic

```python
from pydantic import BaseModel, Field

class ActionItem(BaseModel):
    title: str
    due_date: datetime
    owner: str
    status: Literal["open", "closed"]
```

### with_structured_output()

La forma moderna de obtener outputs estructurados en LangChain:

```python
# Con TypedDict
class UserInfo(TypedDict):
    name: Annotated[str, "Name of the user"]
    age: Annotated[int, "Age of the user"]

llm_structured = llm.with_structured_output(UserInfo)
result = llm_structured.invoke("My name is Daniel, I'm 38 years old")

# Con Pydantic (mas robusto)
class PydanticUserInfo(BaseModel):
    name: str = Field(description="Name of the user")
    age: int = Field(description="Age of the user")

llm_pydantic = llm.with_structured_output(PydanticUserInfo)
```

### Manejo de errores: 3 approaches

| Approach | Cuando usarlo |
|----------|--------------|
| `with_structured_output(include_raw=True)` | Produccion — previene el problema desde el inicio |
| LCEL fix chain (`RunnableLambda`) | JSON malformado ya existente — corrige via LLM |
| `parser.with_fallbacks()` | Integracion en chains LCEL complejas |

> **Archivos:** [`02_structured_outputs/main.py`](02_structured_outputs/main.py)

---

## Leccion 03: De Chains a Runnables y LCEL

### Evolucion de LangChain

LangChain empezo con **Chains** (workflows secuenciales) que hoy estan **deprecated**. El foco actual:

- **LCEL (LangChain Expression Language)**: composicion declarativa.
- **LangGraph**: orquestacion de workflows agentic con estado y ciclos.

### La interfaz Runnable

La interfaz **Runnable** es el bloque base. Todo componente que pueda recibir un input y producir un output la implementa.

```python
runnable.invoke(input)         # Ejecuta una sola vez, sincrono
runnable.batch([i1, i2, i3])   # Ejecuta en paralelo sobre una lista
runnable.stream(input)         # Devuelve tokens/resultados en streaming
await runnable.ainvoke(input)  # Version async
```

### Runnables clave

#### RunnableLambda
Convierte cualquier funcion Python en un Runnable:

```python
from langchain_core.runnables import RunnableLambda

def to_uppercase(text: str) -> str:
    return text.upper()

runnable = RunnableLambda(to_uppercase)
chain = llm | StrOutputParser() | runnable
```

#### RunnableParallel
Ejecuta multiples chains en paralelo y combina resultados:

```python
chain = RunnableParallel({
    "respuesta": llm | StrOutputParser(),
    "longitud": RunnableLambda(lambda x: len(x))
})
```

#### RunnablePassthrough
Pasa el input sin modificarlo (conserva el input original en una cadena):

```python
chain = RunnableParallel({
    "pregunta_original": RunnablePassthrough(),
    "respuesta": llm | StrOutputParser()
})
```

#### RunnableBranch
Routing condicional (if/else para cadenas):

```python
branch = RunnableBranch(
    (lambda x: "codigo" in x, chain_tecnica),
    (lambda x: "poema" in x, chain_creativa),
    chain_generica  # default
)
```

### LCEL: componer como tuberias

LCEL permite componer Runnables con sintaxis tipo pipes:

```python
chain = prompt | llm | output_parser
```

**Beneficios:**
- Evita boilerplate y manejo manual de ejecucion
- Composicion clara y facil de mantener
- El runtime puede optimizar el flujo

### Por que Runnables importan en MLOps

- **Observabilidad**: cualquier Runnable puede ser trazado con LangSmith
- **Composicion declarativa**: defines el pipeline como datos, no logica imperativa
- **Paralelismo gratis**: `batch()` ejecuta en threads automaticamente
- **Streaming sin refactor**: cambiar de `invoke` a `stream` es trivial

> **Archivos:** [`03_runnables_lcel/main.py`](03_runnables_lcel/main.py) | [`03_runnables_lcel/exercise.py`](03_runnables_lcel/exercise.py)

---

## Leccion 04: Introduccion a Sistemas Agentic

### La IA esta cambiando

La IA esta pasando de herramientas reactivas (solo responden) a **sistemas autonomos** que pueden:

- **Tomar decisiones**
- **Usar herramientas** (APIs, bases de datos, buscadores)
- **Mejorar sus respuestas** iterando y corrigiendose

### IA tradicional vs. IA agentic

| IA Tradicional | IA Agentic |
|----------------|------------|
| Responde con texto | Ademas de responder, **hace tareas** |
| GPS que te dice por donde ir | Auto que conduce (planifica, ejecuta, ajusta) |

### Ejemplo: equipo de marketing

Un sistema agentic puede:
1. Leer o escuchar una transcripcion
2. Extraer senales importantes (dolores, necesidades, oportunidades)
3. Resumir hallazgos en lenguaje claro
4. Producir un reporte accionable para ventas

### Que aprenderemos

- Como **darle herramientas** a un modelo para ejecutar funciones externas
- Como hacer que un agente **planifique** pasos
- Como usar **auto-revision (self-reflection)** para mejorar precision
- Como disenar **workflows** practicos y mantenibles
- Como **elegir el enfoque agentic correcto** segun el problema

> **Archivos:** [`04_agentic_fundamentals/`](04_agentic_fundamentals/) *(proximo contenido)*

---

## Estructura del modulo

```
modulo_01_langchain_fundamentals/
├── README.md                          # Este archivo
├── config.yaml                        # Configuracion base del LLM
├── 01_introduccion_langchain/
│   ├── demo.py                        # LLM, Messages, PromptTemplates, FewShot
│   └── exercise.py                    # Chatbot con memoria y few-shot learning
├── 02_structured_outputs/
│   ├── config.yaml
│   └── main.py                        # Parsers, Pydantic, manejo de errores
├── 03_runnables_lcel/
│   ├── config.yaml
│   ├── main.py                        # Runnables, LCEL, composicion
│   └── exercise.py                    # AI Business Advisor (workflow multi-paso)
└── 04_agentic_fundamentals/
    └── (contenido proximo)
```

## Setup

```bash
# Instalar dependencias
uv sync

# Configurar API keys
cp .env.example .env
# Editar .env con tus API keys

# Ejecutar cualquier leccion
python modulo_01_langchain_fundamentals/01_introduccion_langchain/demo.py
```

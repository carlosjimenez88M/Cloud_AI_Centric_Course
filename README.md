# Cloud AI Centric Course

## Objetivo del curso

Ensenar a construir **aplicaciones de IA de produccion** con una progresion clara:

1. **Dominar LangChain y LangGraph** como frameworks fundamentales para trabajar con LLMs
2. **Construir sistemas RAG** (Retrieval-Augmented Generation) y evolucionar hacia **Agentic RAG**
3. **Desplegar agentes en Google Cloud Platform (GCP)** con arquitecturas escalables y listas para produccion

Al finalizar, tendras la capacidad de disenar, implementar y desplegar agentes autonomos que combinan recuperacion de informacion, razonamiento y ejecucion de herramientas.

---

## Estructura del curso

```
Cloud_AI_Centric_Course/
│
├── modulo_01_langchain_fundamentals/    # Fundamentos de LangChain
│   ├── 01_introduccion_langchain/       # LLMs, Messages, Prompts, Few-Shot
│   ├── 02_structured_outputs/           # Pydantic, Output Parsers, Function Calling
│   ├── 03_runnables_lcel/               # Runnables, LCEL, composicion de pipelines
│   └── 04_agentic_fundamentals/         # Intro a sistemas agentic
│
├── modulo_02_rag/                       # RAG y Agentic RAG
│   ├── 01_embeddings_vectorstores/      # Embeddings, vector databases (FAISS, Chroma)
│   ├── 02_retrieval_chains/             # Retrieval chains y document loaders
│   ├── 03_rag_avanzado/                 # Re-ranking, hybrid search, multi-query
│   └── 04_agentic_rag/                  # RAG con agentes autonomos
│
├── modulo_03_langgraph/                 # LangGraph: orquestacion de agentes
│   ├── 01_grafos_y_estado/              # StateGraph, nodos, edges
│   ├── 02_tool_calling/                 # Tools, function calling, agentes reactivos
│   ├── 03_workflows_complejos/          # Branching, ciclos, human-in-the-loop
│   └── 04_multi_agent/                  # Sistemas multi-agente
│
├── modulo_04_gcp_deployment/            # Despliegue en Google Cloud Platform
│   ├── 01_cloud_run/                    # Containerizacion y Cloud Run
│   ├── 02_vertex_ai/                    # Vertex AI para modelos y endpoints
│   ├── 03_cloud_storage_bigquery/       # Almacenamiento y datos para RAG
│   └── 04_arquitectura_produccion/      # Observabilidad, CI/CD, escalabilidad
│
├── src/agentic_ai/                      # Paquete reutilizable del curso
│   ├── agents/                          # Implementaciones de agentes
│   ├── chains/                          # Chains y pipelines
│   ├── config/                          # Configuracion centralizada
│   ├── graphs/                          # Grafos LangGraph
│   ├── prompts/                         # Templates de prompts
│   └── tools/                           # Definiciones de herramientas
│
├── pyproject.toml
├── .env.example
└── README.md
```

---

## Modulos

### Modulo 01: Fundamentos de LangChain
> **Estado: En progreso**

La base de todo lo que viene despues. Cubre la interaccion con LLMs, manejo de mensajes, prompt engineering, structured outputs, la interfaz Runnable, LCEL y la introduccion a sistemas agentic.

[Ver contenido del modulo](modulo_01_langchain_fundamentals/README.md)

### Modulo 02: RAG y Agentic RAG
> **Estado: Proximo**

Desde embeddings y vector stores hasta sistemas RAG completos que combinan retrieval con agentes autonomos capaces de decidir cuando y como buscar informacion.

### Modulo 03: LangGraph — Orquestacion de Agentes
> **Estado: Proximo**

Construccion de workflows complejos con grafos de estado, tool calling, ciclos de razonamiento, human-in-the-loop y sistemas multi-agente.

### Modulo 04: Despliegue en GCP
> **Estado: Proximo**

Llevar los agentes a produccion usando Cloud Run, Vertex AI, Cloud Storage y BigQuery. Arquitecturas escalables con observabilidad y CI/CD.

---

## Requisitos previos

- Python 3.11+
- Conocimientos basicos de Python y programacion orientada a objetos
- Cuenta de OpenAI (API key)
- Cuenta de Google Cloud Platform (para modulo 04)

## Setup rapido

```bash
# 1. Clonar el repositorio
git clone git@github.com:carlosjimenez88M/Cloud_AI_Centric_Course.git
cd Cloud_AI_Centric_Course

# 2. Instalar dependencias con uv
uv sync

# 3. Configurar variables de entorno
cp .env.example .env
# Editar .env con tus API keys

# 4. Ejecutar la primera leccion
python modulo_01_langchain_fundamentals/01_introduccion_langchain/demo.py
```

## Stack tecnologico

| Tecnologia | Uso |
|------------|-----|
| **LangChain** | Framework base para LLMs |
| **LangGraph** | Orquestacion de agentes con estado |
| **LangSmith** | Observabilidad y tracing |
| **OpenAI** | Proveedor de LLMs |
| **Pydantic** | Validacion de schemas |
| **FAISS / Chroma** | Vector databases para RAG |
| **Google Cloud Platform** | Infraestructura de despliegue |
| **uv** | Gestion de dependencias Python |

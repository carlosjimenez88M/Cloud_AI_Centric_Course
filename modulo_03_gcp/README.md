# Módulo 03 — Vertex AI: LLMs, RAG y Orquestación con Google Cloud

Módulo práctico del curso **Cloud AI-Centric** que demuestra cómo construir aplicaciones
de IA generativa de producción usando Google Cloud Platform.

---

## Objetivos

| Script | Concepto | Tecnología |
|--------|----------|-----------|
| `01_role_base.py` | Role-Based Prompting | `google-genai` + Vertex AI |
| `02_rag_pipeline.py` | RAG Pipeline (4 pasos) | GCS + ChromaDB + LangChain |
| `03_orchestration.py` | Orquestación profunda | LangGraph + Chaining + Routing |

---

## Arquitectura del Módulo

```
modulo_03_gcp/
├── 01_role_base.py          ← Launcher: Los 4 Fantásticos + Arjona
├── 02_rag_pipeline.py       ← Launcher: RAG completo (Steps 1-4)
├── 03_orchestration.py      ← Launcher: Orquestación con routing
├── config.yaml              ← Configuración central (modelo, RAG, embeddings)
├── pyproject.toml           ← Dependencias y paquetes instalables
│
└── src/
    ├── .env                 ← PROJECT_ID, GCP_API_KEY, LOCATION
    ├── database/
    │   └── Anónimo Las Mil y Una Noches.pdf   ← Corpus para RAG
    │
    ├── shared/              ← Paquete de utilidades compartidas
    │   ├── logger.py        ← Logger coloreado con Rich
    │   └── config_loader.py ← Carga config.yaml + .env
    │
    ├── 01_role_base/        ← Role-Based Prompting (Los 4 Fantásticos + Arjona)
    │   ├── personas.py      ← Definición de personas y etiquetas
    │   ├── tasks.py         ← Tareas temáticas (4 TODOs)
    │   ├── engine.py        ← PersonaEngine: llamadas a Vertex AI + métricas
    │   └── main.py          ← Orquestación y ranking final
    │
    ├── 02_rag/              ← Pipeline RAG modular (4 steps)
    │   ├── step1_ingest.py  ← Sube PDF a Google Cloud Storage
    │   ├── step2_split.py   ← Divide en chunks semánticos
    │   ├── step3_embed.py   ← Embeddings (Vertex AI) + ChromaDB
    │   └── step4_agent.py   ← Agente LangGraph ReAct
    │
    └── 03_orchestration/    ← Orquestación profunda
        ├── prompts.py       ← Plantillas por tipo de análisis
        ├── chains.py        ← Cadenas LangChain especializadas
        ├── router.py        ← Router LLM (clasifica intención)
        ├── graph.py         ← Grafo LangGraph (Router→RAG→Chain→Síntesis)
        └── main.py          ← Entrada del módulo de orquestación
```

---

## Requisitos Previos

### 1. Autenticación GCP

```bash
# Iniciar sesión con Application Default Credentials
gcloud auth application-default login

# Configurar el proyecto activo
gcloud config set project mlops-practices-wb
```

### 2. Variables de entorno

El archivo `src/.env` debe contener:

```env
PROJECT_ID=mlops-practices-wb
LOCATION=us-central1
GCP_API_KEY=<tu-api-key>
```

### 3. Instalar dependencias

```bash
cd modulo_03_gcp
uv sync
```

---

## Ejecución

> **Todos los comandos se ejecutan desde `modulo_03_gcp/`**

### Lección 1 — Role-Based Prompting

```bash
uv run 01_role_base.py
```

Ejecuta los **4 TODOs** con los **6 personajes** (6 × 4 = 24 llamadas a la API):

| TODO | Tarea |
|------|-------|
| 1 | Análisis de fortalezas y debilidades |
| 2 | Rival más probable por personaje |
| 3 | Canción estilo Ricardo Arjona para la batalla |
| 4 | Mejores escenarios tropicales para cada batalla |

**Personas definidas:**
- Reed Richards — Mr. Fantástico
- Ben Grimm — La Cosa
- Johnny Storm — La Antorcha Humana
- Sue Storm — Mujer Invisible
- Ricardo Arjona
- Mix (síntesis de los 5)

**Duración estimada:** 3-5 minutos

---

### Lección 2 — RAG Pipeline

```bash
uv run 02_rag_pipeline.py
```

Ejecuta el pipeline de 4 pasos sobre **Las Mil y Una Noches**:

```
STEP 1 — Ingest   : Sube el PDF a gs://mlops-practices-wb-cap2-end_to_end/modulo_03_rag/
STEP 2 — Split    : Carga y divide en chunks (1000 chars, 200 overlap)
STEP 3 — Embed    : Genera embeddings con text-embedding-004 → ChromaDB
STEP 4 — Agent    : Agente LangGraph ReAct responde 4 preguntas demo
```

**Notas:**
- `config.yaml → rag.max_pages: 150` limita el índice a 150 páginas para demos rápidos.
- Para indexar el libro completo (4865 páginas, ~16 min): cambia a `max_pages: 0`.
- Idempotente: si el PDF ya está en GCS y ChromaDB ya tiene vectores, los pasos se omiten.

**Duración estimada (demo):** 2-3 minutos

---

### Lección 3 — Orquestación Profunda

```bash
# Requiere que el STEP 3 del RAG ya se haya ejecutado
uv run 03_orchestration.py
```

Pipeline de orquestación con **routing dinámico** y **chaining**:

```
Pregunta
  └─► [Router LLM]     → clasifica: story | character | philosophy | creative
       └─► [RAG]        → recupera top-5 fragmentos del corpus
            └─► [Chain] → análisis especializado (4 cadenas distintas)
                 └─► [Synthesizer] → integra y enriquece la respuesta final
```

**Cadenas disponibles:**
| Ruta | Cadena | Descripción |
|------|--------|-------------|
| `story` | Análisis narrativo | Arco, elementos fantásticos, cuentos anidados |
| `character` | Análisis de personajes | Arquetipo, motivaciones, dinámica de poder |
| `philosophy` | Reflexión filosófica | Moral, pensamiento islámico, vigencia actual |
| `creative` | Generación creativa | Poemas, monólogos, adaptaciones |

**Duración estimada:** 1-2 minutos

---

## Configuración (`config.yaml`)

```yaml
model:
  name: gemini-2.5-flash-lite   # Cambiar a gemini-2.5-pro para mayor calidad
  thinking_budget: 0             # 0 = desactivado (flash-lite/flash)
  temperature: 0.7
  max_output_tokens: 2000

embedding:
  model: text-embedding-004

rag:
  bucket: mlops-practices-wb-cap2-end_to_end
  bucket_prefix: modulo_03_rag/
  chunk_size: 1000
  chunk_overlap: 200
  top_k: 5
  max_pages: 150              # 0 = sin límite
```

### Cambiar de modelo

| Modelo | Velocidad | Calidad | `thinking_budget` |
|--------|-----------|---------|-------------------|
| `gemini-2.5-flash-lite` | ★★★ | ★★ | `0` (desactivado) |
| `gemini-2.5-flash` | ★★ | ★★★ | `0` (desactivado) |
| `gemini-2.5-pro` | ★ | ★★★★ | `>= 1024` (requerido) |

---

## Dónde ver la base vectorial en GCP

Después de ejecutar la Lección 2 con `sync_chroma_gcs: true` en `config.yaml`, la base
vectorial ChromaDB queda sincronizada en Google Cloud Storage.

Para verla:

1. Abre la [GCS Console](https://console.cloud.google.com/storage/browser)
2. Navega al bucket: `mlops-practices-wb-cap2-end_to_end`
3. Entra a la carpeta: `modulo_03_rag/chroma_db/`

Allí encontrarás los archivos internos de ChromaDB (SQLite + segmentos HNSW) que
conforman el índice vectorial del corpus de Las Mil y Una Noches.

---

## Paquetes instalados

Solo el paquete `shared` se instala como paquete Python del entorno virtual.
Los módulos `01_role_base`, `02_rag` y `03_orchestration` se cargan mediante
`sys.path` desde los launchers de la raíz, lo que permite usar nombres de carpeta
con prefijos numéricos (no válidos como identificadores Python):

```python
from shared.logger import get_logger
from shared.config_loader import load_config
```

---

## Referencias

- [Vertex AI Generative AI](https://cloud.google.com/vertex-ai/generative-ai/docs)
- [google-genai SDK](https://googleapis.github.io/python-genai/)
- [LangChain Google Vertex AI](https://python.langchain.com/docs/integrations/chat/google_vertex_ai_palm/)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [ChromaDB](https://docs.trychroma.com/)
- [Diseño de Prompts en Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/prompt-design-strategies)

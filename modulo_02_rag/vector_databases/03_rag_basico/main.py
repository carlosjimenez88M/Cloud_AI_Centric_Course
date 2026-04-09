"""
Módulo 02 - Lección 03: RAG Básico — Retrieval-Augmented Generation

¿Por qué esta lección importa?
─────────────────────────────
  Los LLMs tienen dos limitaciones fundamentales:
    1. Conocimiento congelado en la fecha de entrenamiento
    2. Alucinación: generan texto plausible aunque sea incorrecto

  RAG (Retrieval-Augmented Generation) resuelve ambos problemas:
  → Primero RECUPERA documentos relevantes de tu base de conocimiento
  → Luego le PASA esos documentos al LLM como contexto
  → El LLM GENERA una respuesta fundamentada en esos documentos reales

  El resultado: respuestas actualizadas, verificables y con fuente,
  sin necesidad de reentrenar el modelo.

Arquitectura de un pipeline RAG:

  ┌─────────────────── FASE DE INDEXACIÓN (offline, una vez) ──────────────┐
  │  Documentos → Chunking → Embeddings → ChromaDB                         │
  └────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────── FASE DE CONSULTA (online, por request) ─────────────┐
  │  Query → Embed query → ChromaDB (top-k docs) → Prompt + contexto → LLM │
  └────────────────────────────────────────────────────────────────────────┘

Temas cubiertos:
  1. El problema de la alucinación — LLM sin contexto
  2. Indexación — chunking + embeddings + ChromaDB
  3. Retrieval — recuperar los chunks más relevantes
  4. Generation — prompting con contexto y LCEL chain
  5. Comparación con/sin RAG — la diferencia en práctica
"""

#######################
# ---- Libraries ---- #
#######################

import yaml
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from modulo_02_vector_databases.shared.logger import get_logger

SCRIPT_DIR = Path(__file__).parent

###########################
# ---- Logger Design ---- #
###########################

logger = get_logger(__name__)

##########################
# ---- Load Config ---- #
##########################

try:
    with open(SCRIPT_DIR / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.info(
        f"Config cargado — "
        f"llm: {config['llm_model']}, "
        f"chunk_size: {config['chunk_size']}, "
        f"retrieval_k: {config['retrieval_k']}"
    )
except FileNotFoundError:
    logger.error("No se encontró config.yaml")
    raise
except yaml.YAMLError as e:
    logger.error(f"Error al parsear config.yaml: {e}")
    raise

######################
# ---- Call API ---- #
######################

if not load_dotenv():
    logger.warning("No se encontró .env — buscando claves en el sistema")

try:
    llm = ChatOpenAI(model=config["llm_model"], max_tokens=config["max_tokens"])
    embeddings = OpenAIEmbeddings(model=config["embedding_model"])
    logger.info(f"LLM y embeddings inicializados: {config['llm_model']} / {config['embedding_model']}")
except Exception as e:
    logger.error(f"Error al inicializar modelos: {e}")
    raise RuntimeError("Verifica tu OPENAI_API_KEY en el archivo .env") from e


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 1: El problema de la alucinación
# ══════════════════════════════════════════════════════════════════════════════
#
# Los LLMs generan texto probable dado el contexto. Cuando no saben algo,
# no dicen "no sé" — generan texto que suena correcto pero puede ser falso.
# Esto es alucinación.
#
# El LLM no miente intencionalmente: su arquitectura es estadística.
# Cuando le preguntas algo fuera de su conocimiento, interpola a partir
# de patrones aprendidos. El resultado suena creíble pero puede ser incorrecto.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  BLOQUE 1: El problema — LLM sin contexto propio")
print("═" * 70)

# Pregunta sobre datos ficticios específicos que el LLM no puede conocer
PREGUNTA_DEMO = (
    "¿Cuáles fueron los resultados financieros del proyecto Cóndor Digital "
    "de InnovaLatam en el segundo trimestre de 2025? ¿Cuántos usuarios activos "
    "alcanzaron y cuál fue el margen de ganancia?"
)

logger.info("Consultando LLM sin contexto (modo alucinación)...")

try:
    respuesta_sin_rag = llm.invoke(PREGUNTA_DEMO)
    print(f"\n  Pregunta: {PREGUNTA_DEMO[:80]}...")
    print(f"\n  Respuesta del LLM (sin contexto):")
    print(f"  {'─' * 65}")
    print(f"  {respuesta_sin_rag.content[:400]}...")
    print(
        "\n  ⚠️  Esta respuesta puede ser completamente inventada."
        "\n     El LLM no tiene acceso a este dato específico, pero puede"
        "\n     generar una respuesta que suene convincente y creíble."
    )
except Exception as e:
    logger.error(f"Error al consultar el LLM: {e}")
    raise


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 2: Indexación — Preparar la base de conocimiento
# ══════════════════════════════════════════════════════════════════════════════
#
# Para que el LLM pueda responder con información real, necesitamos:
#   1. Tener los documentos con la información
#   2. Dividirlos en chunks manejables
#   3. Generar embeddings y almacenarlos en ChromaDB
#
# Esta fase se hace OFFLINE — una sola vez cuando cambia el conocimiento.
# En producción, se haría con un pipeline de ingesta automatizado.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  BLOQUE 2: Indexación — Construir la base de conocimiento")
print("═" * 70)

# Documento ficticio pero específico — el LLM no puede conocer estos datos
DOCUMENTO_INNOVALATAM = """
InnovaLatam — Reporte Trimestral Q2 2025

Proyecto Cóndor Digital: Resultados del Segundo Trimestre 2025

El proyecto Cóndor Digital alcanzó en Q2 2025 un total de 847,000 usuarios
activos mensuales (MAU), representando un crecimiento del 34% respecto al
trimestre anterior. La tasa de retención a 30 días se situó en el 71%.

Resultados financieros Q2 2025:
- Ingresos totales: USD 12.4 millones
- Margen bruto: 68%
- Margen de ganancia neto: 23%
- Costo de adquisición de cliente (CAC): USD 14.7
- Valor de vida del cliente (LTV): USD 187.3
- Ratio LTV/CAC: 12.7x (benchmark industria: 3x)

Mercados principales por usuarios activos:
- Brasil: 312,000 usuarios (37% del total)
- México: 241,000 usuarios (28% del total)
- Colombia: 156,000 usuarios (18% del total)
- Argentina: 89,000 usuarios (11% del total)
- Otros LATAM: 49,000 usuarios (6% del total)

Hitos del trimestre:
- Lanzamiento del módulo de pagos en tiempo real con integración PIX (Brasil)
- Alianza estratégica con Banco Andino para distribución en Colombia y Perú
- Certificación ISO 27001 completada en junio 2025
- Expansión del equipo técnico: de 45 a 78 ingenieros

Proyección Q3 2025:
- Meta MAU: 1.1 millones de usuarios
- Ingreso proyectado: USD 16-18 millones
- Nuevos mercados: Chile y Uruguay
"""

DOCUMENTO_CONTEXTO_EMPRESA = """
InnovaLatam: Perfil Corporativo

InnovaLatam es una empresa de tecnología financiera (fintech) fundada en
Bogotá, Colombia en 2019. Opera en 7 países de América Latina y cuenta con
más de 200 empleados a nivel regional.

Su producto principal, Cóndor Digital, es una plataforma de pagos y servicios
financieros B2C diseñada para la población no bancarizada y sub-bancarizada
de América Latina. El 64% de sus usuarios nunca tuvo acceso previo a servicios
financieros formales.

La empresa levantó una Serie B de USD 45 millones en octubre de 2024, liderada
por fondos de venture capital de Silicon Valley y Europa, con participación de
fondos regionales como Kaszek Ventures y Monashees.

CEO: Valentina Torres (ex-Rappi, MBA Wharton)
CTO: Diego Fuentes (ex-Google, Bogotá)
Sede principal: Bogotá, Colombia
Oficinas: Ciudad de México, São Paulo, Lima
"""

logger.info("Chunkeando documentos...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=config["chunk_size"],
    chunk_overlap=config["chunk_overlap"],
)

# split_text retorna strings; los convertimos a Document con metadatos
chunks_reporte = splitter.split_text(DOCUMENTO_INNOVALATAM)
chunks_perfil  = splitter.split_text(DOCUMENTO_CONTEXTO_EMPRESA)

docs_indexados = [
    Document(page_content=chunk, metadata={"fuente": "reporte_q2_2025", "empresa": "InnovaLatam"})
    for chunk in chunks_reporte
] + [
    Document(page_content=chunk, metadata={"fuente": "perfil_corporativo", "empresa": "InnovaLatam"})
    for chunk in chunks_perfil
]

logger.info(
    f"Documentos preparados: "
    f"{len(chunks_reporte)} chunks del reporte + {len(chunks_perfil)} del perfil"
)

print(f"\n  Reporte original: {len(DOCUMENTO_INNOVALATAM):,} chars → {len(chunks_reporte)} chunks")
print(f"  Perfil original:  {len(DOCUMENTO_CONTEXTO_EMPRESA):,} chars → {len(chunks_perfil)} chunks")
print(f"  Total chunks a indexar: {len(docs_indexados)}")

logger.info("Creando vectorstore en ChromaDB...")

try:
    vectorstore = Chroma.from_documents(
        documents=docs_indexados,
        embedding=embeddings,
        collection_name="innovalatam_q2_2025",
    )
    logger.info(f"Vectorstore creado: {vectorstore._collection.count()} chunks indexados")
except Exception as e:
    logger.error(f"Error al crear vectorstore: {e}")
    raise

print(f"  Chunks en ChromaDB: {vectorstore._collection.count()}")


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 3: Retrieval — Recuperar chunks relevantes
# ══════════════════════════════════════════════════════════════════════════════
#
# El retriever convierte la vectorstore en un componente de cadena LCEL.
# Cuando recibe una query:
#   1. Genera el embedding de la query
#   2. Busca los k chunks más similares en ChromaDB
#   3. Retorna los Document como lista
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  BLOQUE 3: Retrieval — ¿Qué recupera el sistema?")
print("═" * 70)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": config["retrieval_k"]},
)

logger.info(f"Recuperando chunks para: '{PREGUNTA_DEMO[:50]}...'")

try:
    chunks_recuperados = retriever.invoke(PREGUNTA_DEMO)
    print(f"\n  Query: '{PREGUNTA_DEMO[:70]}...'")
    print(f"  Chunks recuperados (top {config['retrieval_k']}):\n")
    for i, doc in enumerate(chunks_recuperados, 1):
        fuente = doc.metadata.get("fuente", "?")
        print(f"  [{i}] Fuente: {fuente}")
        print(f"       {doc.page_content[:150].strip()}...")
        print()
except Exception as e:
    logger.error(f"Error en retrieval: {e}")
    raise


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 4: Generation — El pipeline RAG completo con LCEL
# ══════════════════════════════════════════════════════════════════════════════
#
# Construimos la cadena RAG con LCEL (LangChain Expression Language):
#
#   Query
#     │
#     ├──► retriever ──► format_docs ──────────────────────► "context"
#     │                                                          │
#     └──────────────────────────────────────────────────────► "question"
#                                                               │
#                                                         RAG_PROMPT
#                                                               │
#                                                             llm
#                                                               │
#                                                       StrOutputParser
#                                                               │
#                                                          respuesta
#
# La instrucción "usa SOLO el contexto" es crítica para evitar alucinación.
# Sin ella, el LLM combinará su conocimiento general con el contexto y puede
# inventar información adicional.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  BLOQUE 4: Pipeline RAG completo (LCEL)")
print("═" * 70)


def format_docs(docs: list[Document]) -> str:
    """
    Concatena los documentos recuperados en un único string de contexto.

    Esta función es el "pegamento" entre el retriever y el prompt.
    Separamos cada documento con doble salto de línea para que el LLM
    pueda distinguir dónde termina uno y comienza el siguiente.
    """
    return "\n\n".join(doc.page_content for doc in docs)


# Prompt de RAG: la instrucción de ceñirse al contexto es FUNDAMENTAL
RAG_PROMPT = ChatPromptTemplate.from_template(
    """Eres un analista financiero que responde preguntas basándote ÚNICAMENTE
en la información proporcionada en el contexto. No uses conocimiento externo.

Si la respuesta no está en el contexto, responde exactamente:
"No tengo esa información en los documentos disponibles."

Contexto:
{context}

Pregunta: {question}

Respuesta:"""
)

# Cadena LCEL del pipeline RAG
# RunnablePassthrough() simplemente pasa la query sin modificarla
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)

logger.info("Ejecutando pipeline RAG completo...")

try:
    respuesta_rag = rag_chain.invoke(PREGUNTA_DEMO)
    print(f"\n  Pregunta: {PREGUNTA_DEMO[:80]}...")
    print(f"\n  Respuesta RAG (con contexto real):")
    print(f"  {'─' * 65}")
    print(f"  {respuesta_rag}")
except Exception as e:
    logger.error(f"Error en el pipeline RAG: {e}")
    raise


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 5: Comparación directa — con RAG vs sin RAG
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  BLOQUE 5: Comparación — Con RAG vs Sin RAG")
print("═" * 70)

preguntas_comparacion = [
    "¿Cuál fue el margen de ganancia neto de Cóndor Digital en Q2 2025?",
    "¿En qué países opera InnovaLatam y cuántos usuarios tiene en cada uno?",
    "¿Quién es el CEO de InnovaLatam y cuál es su experiencia profesional?",
]

for pregunta in preguntas_comparacion:
    print(f"\n  📋 Pregunta: '{pregunta}'")
    print(f"  {'─' * 65}")

    # Sin RAG
    try:
        sin_rag_content = str(llm.invoke(pregunta).content)
        print(f"\n  ❌ Sin RAG: {sin_rag_content[:200].strip()}...")
    except Exception as e:
        logger.error(f"Error sin RAG: {e}")
        raise

    # Con RAG
    try:
        con_rag = rag_chain.invoke(pregunta)
        print(f"\n  ✅ Con RAG: {con_rag[:200].strip()}")
    except Exception as e:
        logger.error(f"Error con RAG: {e}")
        raise

    print()

print("\n" + "═" * 70)
logger.info(
    "Lección 03 completada. "
    "El pipeline RAG básico funciona. "
    "Siguiente: Lección 04 — técnicas avanzadas para mejorar el retrieval."
)
print("═" * 70 + "\n")

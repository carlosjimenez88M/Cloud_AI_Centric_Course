"""
Módulo 02 - Lección 03: Exercise — RAG sobre Cambio Climático en Latinoamérica

¿Qué construimos aquí?
─────────────────────
  Un sistema RAG completo sobre un corpus de documentos de política climática.
  El objetivo es practicar el pipeline end-to-end con un tema de relevancia
  regional: cómo América Latina enfrenta el cambio climático.

  Lo que hace este ejercicio diferente al demo:
    1. Corpus más rico y diverso (3 documentos con perspectivas distintas)
    2. Preguntas de distintas naturalezas (factual, comparativa, de síntesis)
    3. Mostrar qué chunks se usaron para responder (trazabilidad)
    4. Manejar el caso donde el contexto NO tiene la respuesta

  La trazabilidad (saber qué chunk usó el LLM) es fundamental en producción:
  permite auditar respuestas, detectar errores y mejorar el retrieval.
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

######################
# ---- Call API ---- #
######################

if not load_dotenv():
    logger.warning("No se encontró .env")

try:
    with open(SCRIPT_DIR / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    llm        = ChatOpenAI(model=config["llm_model"], max_tokens=config["max_tokens"])
    embeddings = OpenAIEmbeddings(model=config["embedding_model"])
    logger.info(f"Modelos listos: {config['llm_model']} / {config['embedding_model']}")
except Exception as e:
    logger.error(f"Error al inicializar: {e}")
    raise RuntimeError("Verifica tu OPENAI_API_KEY y config.yaml") from e


# ══════════════════════════════════════════════════════════════════════════════
# Corpus de Documentos — Cambio Climático en América Latina
# ══════════════════════════════════════════════════════════════════════════════

DOCUMENTOS_CLIMA = [
    Document(
        page_content="""
Informe de Vulnerabilidad Climática de América Latina 2025

América Latina concentra el 8% de la población mundial pero ya sufre el
impacto desproporcionado del cambio climático. La región alberga el 40% de la
biodiversidad del planeta, incluyendo la Amazonia, el sistema de arrecifes
mesoamericanos y los glaciares andinos.

Impactos actuales documentados:
- Los glaciares andinos han perdido el 30-50% de su volumen desde 1980.
  Países como Bolivia, Perú y Ecuador dependen de ellos para agua potable
  y generación hidroeléctrica. El glaciar Chacaltaya en Bolivia desapareció
  completamente en 2009.

- El fenómeno El Niño se ha intensificado: las sequías en Centroamérica
  y el norte de Sudamérica son más frecuentes y prolongadas. En 2023-2024,
  Colombia, Venezuela y partes de Brasil registraron las sequías más graves
  en 50 años.

- El nivel del mar sube a razón de 3.7mm por año en el Caribe, amenazando
  ciudades costeras como Cartagena de Indias, Ciudad de Panamá y sectores
  de Buenos Aires y Montevideo.

- La Amazonia brasileña y boliviana registró en 2023 la temporada de incendios
  más severa en tres décadas: 3.2 millones de hectáreas quemadas.

Proyecciones 2050 sin acción climática:
- Pérdida del 11-17% del PIB regional anual
- 35 millones de migrantes climáticos internos
- Desaparición del 60% de los arrecifes coralinos del Caribe
- Reducción del 25% en la productividad agrícola del Cono Sur
""",
        metadata={"fuente": "informe_vulnerabilidad_2025", "region": "América Latina", "tipo": "diagnóstico"},
    ),
    Document(
        page_content="""
Políticas de Acción Climática en América Latina: Avances 2020-2025

Brasil — Fondo Amazonia 2.0:
El gobierno brasileño relanzó en 2023 el Fondo Amazonia con una capitalización
de USD 3,000 millones aportados por Noruega, Alemania y la Unión Europea.
El fondo financia la prevención de deforestación, restauración y economías
forestales sostenibles para comunidades indígenas. La deforestación cayó un
50% entre agosto 2022 y agosto 2023 respecto al año anterior.

Colombia — Tasa de Carbono:
Colombia opera desde 2017 el impuesto al carbono más ambicioso de América
Latina: USD 5 por tonelada de CO2 emitida por combustibles fósiles. Los
ingresos financian el Fondo para la Sostenibilidad Ambiental, que ha invertido
USD 800 millones en proyectos de restauración y transición energética.

Chile — Transformación Energética:
Chile tiene como meta el 70% de energía renovable para 2030. A 2024, el
45% de la matriz eléctrica ya es renovable (principalmente solar en el
desierto de Atacama y eólica en la Patagonia). El costo de la energía solar
cayó un 89% entre 2010 y 2024, haciendo de Chile el país con energía solar
más barata del hemisferio.

México — Transporte:
Ciudad de México opera la flota de autobuses eléctricos más grande de América
Latina: 3,200 unidades a 2024. El Metro de la Ciudad de México se alimenta
100% de energía renovable desde 2022.

Argentina — Hidrógeno Verde:
Argentina tiene el mayor potencial de producción de hidrógeno verde de América
del Sur, aprovechando los vientos patagónicos. En 2024 exportó los primeros
cargamentos de hidrógeno verde a Europa a través del proyecto HyChico en
Comodoro Rivadavia.
""",
        metadata={"fuente": "politicas_climaticas_2025", "region": "América Latina", "tipo": "política"},
    ),
    Document(
        page_content="""
Financiamiento Climático para América Latina: Brechas y Oportunidades

La región necesita USD 350-400 mil millones anuales para cumplir sus compromisos
del Acuerdo de París. Actualmente recibe apenas USD 55 mil millones en
financiamiento climático (2024), lo que representa una brecha del 85%.

Fuentes de financiamiento climático activas (2024):
- Banco Interamericano de Desarrollo (BID): USD 14 mil millones aprobados en 2024
  con el 35% destinado a proyectos de adaptación y el 65% a mitigación.
- CAF (Banco de Desarrollo de América Latina): USD 8.5 mil millones en 2024.
  CAF es el mayor financiador climático multilateral para la región.
- Green Climate Fund (GCF): La región accedió a USD 1.2 mil millones en 2023-2024.
  Brasil y Colombia son los mayores receptores.
- Bonos verdes: El mercado de bonos verdes en América Latina creció 280% entre
  2019 y 2024, con Chile, Brasil y México liderando la emisión.

Obstáculos estructurales:
1. Capacidad técnica limitada para formular proyectos elegibles
2. Costos de transacción altos para economías pequeñas (ej. Bolivia, Paraguay)
3. Reglas de los fondos internacionales diseñadas para contextos anglosajones
4. Incertidumbre regulatoria en algunos mercados (Argentina, Venezuela)

Mecanismos innovadores emergentes:
- Deuda por naturaleza (debt-for-nature swaps): Ecuador canjeó USD 1,600 millones
  de deuda soberana en 2023 por compromisos de conservación de las Galápagos.
- Mercados de carbono voluntarios: Colombia y Brasil lideran la región con
  proyectos certificados REDD+ y VCS que generan créditos de carbono.
""",
        metadata={"fuente": "financiamiento_climatico_2025", "region": "América Latina", "tipo": "financiamiento"},
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline RAG con Trazabilidad de Fuentes
# ══════════════════════════════════════════════════════════════════════════════

def construir_rag(
    documentos: list[Document],
    modelo_embeddings: OpenAIEmbeddings,
    modelo_llm: ChatOpenAI,
    chunk_size: int,
    chunk_overlap: int,
    retrieval_k: int,
) -> tuple:
    """
    Construye el pipeline RAG completo con soporte de trazabilidad.

    Retorna (chain_rag, retriever) separados para poder inspeccionar
    qué documentos se recuperaron sin ejecutar la generación completa.

    Returns:
        (rag_chain, retriever) — ambos invocables con una query string.
    """
    # Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks: list[Document] = []
    for doc in documentos:
        sub_chunks = splitter.split_text(doc.page_content)
        for chunk_text in sub_chunks:
            chunks.append(Document(
                page_content=chunk_text.strip(),
                metadata=doc.metadata,
            ))

    if not chunks:
        raise ValueError("El chunking no produjo ningún fragmento. Verifica los documentos.")

    logger.info(f"Corpus dividido en {len(chunks)} chunks de tamaño ~{chunk_size}")

    # Indexación
    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=modelo_embeddings,
            collection_name="clima_latam_exercise",
        )
        logger.info(f"ChromaDB indexado: {vectorstore._collection.count()} chunks")
    except Exception as e:
        logger.error(f"Error al indexar en ChromaDB: {e}")
        raise RuntimeError(f"Fallo en la indexación: {e}") from e

    # Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": retrieval_k})

    # Prompt con instrucción estricta + pedido de fuente
    prompt = ChatPromptTemplate.from_template(
        """Eres un experto en política climática de América Latina.
Responde basándote ÚNICAMENTE en el contexto proporcionado.
Si la información no está en el contexto, di exactamente:
"Esta información no está disponible en los documentos del sistema."

Al final de tu respuesta, indica entre corchetes qué fuente(s) usaste.

Contexto:
{context}

Pregunta: {question}

Respuesta:"""
    )

    def format_docs_con_fuente(docs: list[Document]) -> str:
        partes = []
        for doc in docs:
            fuente = doc.metadata.get("fuente", "desconocida")
            partes.append(f"[Fuente: {fuente}]\n{doc.page_content}")
        return "\n\n---\n\n".join(partes)

    rag_chain = (
        {"context": retriever | format_docs_con_fuente, "question": RunnablePassthrough()}
        | prompt
        | modelo_llm
        | StrOutputParser()
    )

    return rag_chain, retriever


# ── Construcción del pipeline ─────────────────────────────────────────────────

print("\n" + "═" * 70)
print("  EXERCISE: RAG — Cambio Climático en América Latina")
print("═" * 70)

logger.info("Construyendo pipeline RAG sobre corpus de cambio climático...")

try:
    rag_chain, retriever = construir_rag(
        documentos=DOCUMENTOS_CLIMA,
        modelo_embeddings=embeddings,
        modelo_llm=llm,
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        retrieval_k=config["retrieval_k"],
    )
    logger.info("Pipeline RAG listo")
except (ValueError, RuntimeError) as e:
    logger.error(f"Error al construir el pipeline: {e}")
    raise


# ── Consultas de demostración ─────────────────────────────────────────────────

CONSULTAS = [
    {
        "pregunta": "¿Qué porcentaje de energía renovable tiene Chile y cuáles son sus fuentes principales?",
        "tipo": "factual",
    },
    {
        "pregunta": "¿Cuánto financiamiento climático necesita América Latina y cuánto recibe actualmente?",
        "tipo": "cuantitativa",
    },
    {
        "pregunta": "¿Qué países de América Latina tienen las políticas climáticas más avanzadas y por qué?",
        "tipo": "comparativa/síntesis",
    },
    {
        "pregunta": "¿Cuál es el precio del barril de petróleo en Venezuela en 2025?",
        "tipo": "out-of-context — no debe estar en los documentos",
    },
]

for consulta in CONSULTAS:
    pregunta = consulta["pregunta"]
    tipo     = consulta["tipo"]

    print(f"\n  📋 [{tipo.upper()}]")
    print(f"  Pregunta: '{pregunta}'")
    print(f"  {'─' * 65}")

    # Mostrar qué chunks se recuperaron (trazabilidad)
    try:
        chunks_recuperados = retriever.invoke(pregunta)
        logger.debug(f"Chunks recuperados: {len(chunks_recuperados)}")
        for i, doc in enumerate(chunks_recuperados, 1):
            fuente = doc.metadata.get("fuente", "?")
            logger.debug(f"  Chunk {i} [{fuente}]: {doc.page_content[:60].strip()}...")
    except Exception as e:
        logger.warning(f"Error al recuperar chunks: {e}")

    # Respuesta del pipeline RAG
    try:
        respuesta = rag_chain.invoke(pregunta)
        print(f"\n  Respuesta:\n  {respuesta.strip()}")
    except Exception as e:
        logger.error(f"Error al generar respuesta: {e}")
        raise

    print()

print("\n" + "═" * 70)
logger.info(
    "Exercise 03 completado. "
    "Notaste cómo el sistema responde 'no disponible' cuando la info no está? "
    "Eso es RAG honesto. En la Lección 04, mejoraremos el retrieval."
)
print("═" * 70 + "\n")

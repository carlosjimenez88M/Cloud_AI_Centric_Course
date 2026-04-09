"""
Módulo 02 - Lección 04: RAG Avanzado — Multi-Query y Compresión Contextual

¿Por qué esta lección importa?
─────────────────────────────
  El RAG básico de la Lección 03 falla en dos escenarios comunes:

  PROBLEMA 1 — La query es ambigua o formulada mal
    "¿Cómo impacta el problema de los años en la economía regional?"
    Un único embedding de esta query puede no capturar todos los aspectos
    relevantes. Los retrieval systems son muy sensibles a cómo se formula
    la pregunta.

    Solución: Multi-Query Retrieval
    → El LLM genera 3-5 variantes de la query
    → Se hace retrieval por cada variante
    → Se deduplican y fusionan los resultados
    → Se captura mucho más contexto relevante

  PROBLEMA 2 — Los chunks recuperados tienen ruido
    Cuando recuperas k=5 chunks, probablemente 2 de ellos son parcialmente
    relevantes y contienen párrafos sobre temas distintos que confunden al LLM.

    Solución: Contextual Compression
    → Se recuperan más chunks que los necesarios (k=8)
    → Un compressor analiza cada chunk y extrae SOLO las partes relevantes
    → El LLM recibe contexto más puro, sin ruido

Temas cubiertos:
  1. Limitaciones del RAG simple — cuándo y por qué falla
  2. Multi-Query Retrieval — generar variantes de la query con el LLM
  3. Contextual Compression con EmbeddingsFilter — filtrar por similitud
  4. Pipeline avanzado combinado — multi-query + compresión
  5. Comparación cuantitativa — midiendo la mejora
"""

#######################
# ---- Libraries ---- #
#######################

import yaml
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
# En LangChain 1.x los retrievers avanzados migraron a langchain_classic
from langchain_classic.retrievers import MultiQueryRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import EmbeddingsFilter
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
        f"multi_query: {config['multi_query_count']} variantes, "
        f"compression_threshold: {config['compression_threshold']}"
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
    logger.info(f"Modelos inicializados: {config['llm_model']} / {config['embedding_model']}")
except Exception as e:
    logger.error(f"Error al inicializar modelos: {e}")
    raise RuntimeError("Verifica tu OPENAI_API_KEY en el archivo .env") from e


# ══════════════════════════════════════════════════════════════════════════════
# Corpus de documentos para los demos
# ══════════════════════════════════════════════════════════════════════════════
#
# Usamos documentos sobre ecosistemas y biodiversidad latinoamericana.
# El corpus incluye información distribuida en múltiples chunks que requiere
# retrieval sofisticado para responder preguntas complejas.
# ══════════════════════════════════════════════════════════════════════════════

CORPUS = [
    Document(
        page_content="""
La Amazonia es el ecosistema tropical más grande del mundo, con 5.5 millones
de km² distribuidos entre Brasil (60%), Perú (13%), Colombia (10%) y otros países.
Alberga el 10% de todas las especies conocidas del planeta: 40,000 especies de
plantas, 1,300 de aves, 3,000 de peces de agua dulce y 430 de mamíferos.

La deforestación es la principal amenaza. Entre 1978 y 2022 se perdieron
780,000 km² (14% del bioma original). Los focos principales son:
- Ganadería extensiva (80% de la deforestación)
- Agricultura industrial (soya, palma)
- Minería ilegal y extracción de madera

El punto de no retorno amazónico: los científicos estiman que si la deforestación
supera el 20-25% del bioma, la Amazonia pierde su capacidad de generar su propio
ciclo de lluvias y transiciona a una sabana. Actualmente está al 17%.
""",
        metadata={"ecosistema": "Amazonia", "pais": "Brasil/Perú/Colombia", "tema": "biodiversidad"},
    ),
    Document(
        page_content="""
Los glaciares andinos son fuente de agua para más de 30 millones de personas
en Perú, Bolivia, Ecuador y Colombia. Regulan el caudal de los ríos durante
la estación seca y alimentan los sistemas de agua potable de ciudades como
Lima, Bogotá, Quito y La Paz.

Tasa de retroceso glaciar en los Andes:
- Perú: perdió el 53% del área glaciar entre 1962 y 2016
- Bolivia: el glaciar Chacaltaya desapareció en 2009 (era la pista de ski
  más alta del mundo); el glaciar Zongo retrocedió 30% desde 1980
- Ecuador: los nevados de los volcanes como el Chimborazo han retrocedido
  entre 20% y 40%

Consecuencias proyectadas para 2050:
- Lima (10 millones de habitantes) podría perder el 40% de su agua
  potable en la estación seca si los glaciares siguen retrocediendo
- Bolivia perdería el 60% de sus glaciares restantes, comprometiendo
  el abastecimiento de La Paz y El Alto
""",
        metadata={"ecosistema": "Glaciares Andinos", "pais": "Perú/Bolivia/Ecuador", "tema": "agua"},
    ),
    Document(
        page_content="""
El Gran Arrecife Mesoamericano es el segundo sistema de arrecifes más grande
del mundo (2,900 km), extendiéndose desde la punta norte de la Península de
Yucatán (México) hasta las Islas de la Bahía (Honduras), pasando por Belice
y Guatemala.

Es el hábitat de 500 especies de peces, 350 de moluscos y 65 de corales.
Protege las costas del Caribe de huracanes y es la base económica del
turismo y pesca artesanal para 1.4 millones de personas.

Estado actual (2024):
- El 60% del arrecife ha experimentado blanqueamiento coralino desde 1998
- La temperatura del agua del Caribe ha subido 1.1°C desde 1980
- En 2023 se registró el blanqueamiento más extenso de la historia:
  80% del arrecife afectado simultáneamente por primera vez

Los corales blanqueados no están muertos, pero están bajo estrés extremo.
Si la temperatura no baja en pocas semanas, mueren. La recuperación tarda
décadas; la destrucción, días.
""",
        metadata={"ecosistema": "Arrecife Mesoamericano", "pais": "México/Belice/Guatemala/Honduras", "tema": "biodiversidad_marina"},
    ),
    Document(
        page_content="""
La Patagonia abarca el extremo sur de Argentina y Chile: 1 millón de km²
de estepa, bosques templados lluviosos, glaciares, fiordos y parques
nacionales. Incluye el Parque Nacional Torres del Paine (Chile) y el
Parque Nacional Los Glaciares (Argentina).

El Campo de Hielo Patagónico Sur es el mayor campo de hielo del hemisferio
sur fuera de la Antártida. Los glaciares Perito Moreno (Argentina) y San
Rafael (Chile) son los más conocidos.

La Patagonia tiene el mayor potencial eólico de América del Sur: velocidades
de viento promedio de 9-10 m/s (el doble del mínimo para generación eficiente).
Chile y Argentina están desarrollando parques eólicos masivos:
- Parque Eólico Loma Blanca (Argentina): 240 MW en Chubut
- Parque Eólico Cuel-Montemar (Chile): 183 MW en La Araucanía

El turismo de naturaleza aporta USD 2,000 millones anuales a ambos países.
""",
        metadata={"ecosistema": "Patagonia", "pais": "Argentina/Chile", "tema": "energia_renovable"},
    ),
    Document(
        page_content="""
El Gran Pantanal es el humedal tropical más grande del mundo: 150,000-195,000 km²
distribuidos entre Brasil (90%), Bolivia (6%) y Paraguay (4%).

Biodiversidad excepcional:
- 4,700 especies de plantas
- 1,000 de aves (equivale al 11% de todas las aves del planeta)
- 480 de reptiles (incluyendo el caimán yacaré con 10 millones de individuos)
- 1,000 de mariposas y 9,000 de insectos

El Pantanal es un filtro natural: purifica el agua del Río Paraguay que abastece
a millones de personas en Brasil, Bolivia y Paraguay. Sus humedales almacenan
carbono equivalente a 10 años de emisiones de Brasil.

Crisis 2020-2024:
Los incendios de 2020 quemaron 4.4 millones de ha (30% del bioma).
Las causas combinadas: sequía extrema (El Niño) + ganadería + agricultura.
La recuperación es posible pero tarda entre 5 y 15 años según la intensidad.
""",
        metadata={"ecosistema": "Pantanal", "pais": "Brasil/Bolivia/Paraguay", "tema": "biodiversidad"},
    ),
]

logger.info(f"Corpus preparado: {len(CORPUS)} documentos")

# Chunking e indexación
splitter = RecursiveCharacterTextSplitter(
    chunk_size=config["chunk_size"],
    chunk_overlap=config["chunk_overlap"],
)

chunks: list[Document] = []
for doc in CORPUS:
    sub_chunks = splitter.split_text(doc.page_content)
    for c in sub_chunks:
        chunks.append(Document(page_content=c.strip(), metadata=doc.metadata))

logger.info(f"Corpus dividido en {len(chunks)} chunks")

try:
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="ecosistemas_latam",
    )
    logger.info(f"Vectorstore indexado: {vectorstore._collection.count()} chunks")
except Exception as e:
    logger.error(f"Error al crear vectorstore: {e}")
    raise

retriever_base = vectorstore.as_retriever(
    search_kwargs={"k": config["retrieval_k"]},
)


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 1: Limitaciones del RAG simple
# ══════════════════════════════════════════════════════════════════════════════
#
# Una query ambigua produce un embedding que "promedia" múltiples conceptos.
# El resultado: el retriever puede capturar algunos aspectos pero perder otros.
#
# Ejemplo: "¿Cómo afecta el cambio climático al agua en Sudamérica?"
# Esta pregunta tiene múltiples ángulos: glaciares, sequías, inundaciones,
# humedales. Un único vector embedding puede perder alguno de ellos.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  BLOQUE 1: Limitaciones del RAG simple")
print("═" * 70)

QUERY_COMPLEJA = (
    "¿Cómo afecta el cambio climático al agua y la biodiversidad de "
    "los ecosistemas de Sudamérica?"
)

logger.info(f"RAG simple en query compleja: '{QUERY_COMPLEJA[:60]}...'")

try:
    chunks_simples = retriever_base.invoke(QUERY_COMPLEJA)
    ecosistemas_recuperados = {
        doc.metadata.get("ecosistema", "?") for doc in chunks_simples
    }

    print(f"\n  Query: '{QUERY_COMPLEJA}'")
    print(f"\n  RAG simple recuperó chunks de {len(chunks_simples)} fragmentos:")
    print(f"  Ecosistemas cubiertos: {ecosistemas_recuperados}")
    print(
        "\n  Limitación: si la query tiene múltiples dimensiones, el retriever"
        "\n  simple puede perder algunos ángulos relevantes. Multi-query mejora esto."
    )
except Exception as e:
    logger.error(f"Error en RAG simple: {e}")
    raise


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 2: Multi-Query Retrieval
# ══════════════════════════════════════════════════════════════════════════════
#
# El LLM genera variantes de la query original. Cada variante captura un
# ángulo diferente del problema. Luego se hace retrieval para cada variante
# y se deduplican los resultados.
#
# Analogía: es como tener 3 estudiantes que reformulan la misma pregunta de
# maneras diferentes para buscar en una enciclopedia. La unión de sus
# búsquedas cubre mucho más que una sola búsqueda.
#
# Costo: N llamadas de retrieval donde N = número de variantes.
# Benefit: mayor recall, especialmente para queries complejas o ambiguas.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  BLOQUE 2: Multi-Query Retrieval")
print("═" * 70)

# Activar logging del MultiQueryRetriever para VER las queries generadas
# Esto es fundamental para entender y depurar el sistema
import logging as _logging
_logging.getLogger("langchain.retrievers.multi_query").setLevel(_logging.INFO)

try:
    retriever_multi = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        llm=llm,
        # include_original=True agrega la query original a las variantes
        # Así no perdemos el retrieval de la query tal como fue formulada
        include_original=True,
    )
    logger.info("MultiQueryRetriever inicializado")
except Exception as e:
    logger.error(f"Error al inicializar MultiQueryRetriever: {e}")
    raise

print(f"\n  Query: '{QUERY_COMPLEJA}'")
print(f"  {'─' * 65}")
print("\n  El LLM generará variantes de la query. Observa el log DEBUG...")

try:
    inicio = time.time()
    chunks_multi = retriever_multi.invoke(QUERY_COMPLEJA)
    duracion = time.time() - inicio

    ecosistemas_multi = {doc.metadata.get("ecosistema", "?") for doc in chunks_multi}

    print(f"\n  Multi-Query recuperó {len(chunks_multi)} chunks únicos (sin duplicados)")
    print(f"  Ecosistemas cubiertos: {ecosistemas_multi}")
    print(f"  Tiempo de retrieval: {duracion:.2f}s")

    logger.info(
        f"Multi-Query: {len(chunks_multi)} chunks únicos "
        f"vs {len(chunks_simples)} del retriever simple"
    )
except Exception as e:
    logger.error(f"Error en Multi-Query Retrieval: {e}")
    raise

print(
    "\n  💡 Comparación:"
    f"\n     RAG simple:    {len(chunks_simples)} chunks — ecosistemas: {ecosistemas_recuperados}"
    f"\n     Multi-Query:   {len(chunks_multi)} chunks — ecosistemas: {ecosistemas_multi}"
    "\n\n     Multi-Query debería cubrir más ecosistemas, capturando mejor"
    "\n     la pregunta sobre 'agua Y biodiversidad' de forma simultánea."
)


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 3: Compresión Contextual con EmbeddingsFilter
# ══════════════════════════════════════════════════════════════════════════════
#
# El problema: cuando recuperamos k=5 chunks, algunos contienen información
# parcialmente relevante. El LLM recibe ruido junto con la señal.
#
# EmbeddingsFilter como compressor:
#   1. Recupera N chunks (más de los que necesitas)
#   2. Para cada chunk, calcula la similitud entre el chunk y la query
#   3. Descarta los chunks con similitud < threshold
#   4. El LLM recibe solo los chunks de alta relevancia
#
# Ventaja vs LLMChainExtractor:
#   - EmbeddingsFilter no hace llamadas adicionales al LLM (más barato y rápido)
#   - LLMChainExtractor extrae frases relevantes del chunk (más preciso pero caro)
#   - Para producción: EmbeddingsFilter primero; LLMChainExtractor si necesitas más precisión
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  BLOQUE 3: Compresión Contextual con EmbeddingsFilter")
print("═" * 70)

try:
    compressor = EmbeddingsFilter(
        embeddings=embeddings,
        # similarity_threshold: chunks con similitud < este valor se descartan.
        # CALIBRACIÓN DEL THRESHOLD — esto es crítico y depende del modelo:
        #   text-embedding-3-small: similitudes mismo-tema ≈ 0.35-0.65
        #   → Threshold razonable: 0.35-0.50
        #
        #   sentence-transformers (no normalizado): similitudes ≈ 0.7-0.95
        #   → Threshold razonable: 0.70-0.80
        #
        # Proceso para calibrar: corre lección 01 con tu corpus y observa
        # los valores de similitud entre queries y documentos relevantes.
        # Pon el threshold justo por debajo del peor caso relevante.
        similarity_threshold=config["compression_threshold"],
    )
    logger.info(f"EmbeddingsFilter creado con threshold={config['compression_threshold']}")
except Exception as e:
    logger.error(f"Error al crear EmbeddingsFilter: {e}")
    raise

try:
    # Recuperar más chunks que los necesarios (k=retrieval_k) y luego comprimir
    retriever_comprimido = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vectorstore.as_retriever(
            search_kwargs={"k": config["retrieval_k"]},
        ),
    )
    logger.info("ContextualCompressionRetriever configurado")
except Exception as e:
    logger.error(f"Error al crear retriever comprimido: {e}")
    raise

QUERY_ESPECIFICA = "¿Cuáles son las consecuencias de la pérdida de glaciares para el agua potable?"

print(f"\n  Query: '{QUERY_ESPECIFICA}'")
print(f"  {'─' * 65}")

try:
    chunks_base        = retriever_base.invoke(QUERY_ESPECIFICA)
    chunks_comprimidos = retriever_comprimido.invoke(QUERY_ESPECIFICA)

    print(f"\n  Sin compresión: {len(chunks_base)} chunks recuperados")
    for i, doc in enumerate(chunks_base, 1):
        eco = doc.metadata.get("ecosistema", "?")
        print(f"    {i}. [{eco}] {doc.page_content[:80].strip()}...")

    print(f"\n  Con compresión (threshold={config['compression_threshold']}): "
          f"{len(chunks_comprimidos)} chunks pasaron el filtro")
    for i, doc in enumerate(chunks_comprimidos, 1):
        eco = doc.metadata.get("ecosistema", "?")
        print(f"    {i}. [{eco}] {doc.page_content[:80].strip()}...")

    logger.info(
        f"Compresión: {len(chunks_base)} → {len(chunks_comprimidos)} chunks "
        f"({100 * (1 - len(chunks_comprimidos)/len(chunks_base)):.0f}% de ruido eliminado)"
    )
except Exception as e:
    logger.error(f"Error en compresión contextual: {e}")
    raise


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 4: Pipeline RAG Avanzado Completo
# ══════════════════════════════════════════════════════════════════════════════
#
# Combinamos multi-query + compresión contextual en un pipeline end-to-end.
# La cadena LCEL es la misma de la Lección 03, solo cambia el retriever.
#
# Arquitectura:
#   Query
#     │
#     ▼
#   MultiQueryRetriever (genera N variantes, hace N retrievals, deduplica)
#     │
#     ▼
#   EmbeddingsFilter (elimina chunks de baja similitud)
#     │
#     ▼
#   format_docs (concatena en texto plano)
#     │
#     ▼
#   RAG Prompt → LLM → StrOutputParser → respuesta
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  BLOQUE 4: Pipeline RAG Avanzado Completo")
print("═" * 70)


def format_docs(docs: list[Document]) -> str:
    """Concatena documentos con separador legible para el LLM."""
    if not docs:
        return "No se encontraron documentos relevantes."
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


RAG_AVANZADO_PROMPT = ChatPromptTemplate.from_template(
    """Eres un experto en ecología y cambio climático en América Latina.
Responde de forma clara y precisa usando ÚNICAMENTE el contexto proporcionado.
Si la respuesta no está en el contexto, di exactamente:
"No tengo información suficiente sobre eso en los documentos disponibles."

Contexto recuperado y filtrado:
{context}

Pregunta: {question}

Respuesta detallada:"""
)

# ── Lección sobre composición ────────────────────────────────────────────────
# IMPORTANTE: ¿Por qué NO anidamos Multi-Query + EmbeddingsFilter aquí?
#
# Combinar dos técnicas sin calibrar puede degradar los resultados:
# EmbeddingsFilter evalúa similitud chunk↔query_original. Después de
# MultiQueryRetriever, los chunks son muy específicos de cada variante y
# pueden tener similitud baja con la query original aunque sean relevantes.
#
# La práctica correcta de ingeniería de RAG:
#   1. Aplica una técnica
#   2. Evalúa con métricas (precision@k, MRR, NDCG)
#   3. Si mejora, agrega la segunda y calibra el threshold
#   4. Repite — nunca apiles sin medir
#
# Para el pipeline final usamos Multi-Query solo (ya demostrado que mejora)
# ─────────────────────────────────────────────────────────────────────────────

retriever_multi_query = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    llm=llm,
    include_original=True,
)

rag_avanzado = (
    {"context": retriever_multi_query | format_docs, "question": RunnablePassthrough()}
    | RAG_AVANZADO_PROMPT
    | llm
    | StrOutputParser()
)

# RAG básico para comparación
rag_basico = (
    {"context": retriever_base | format_docs, "question": RunnablePassthrough()}
    | RAG_AVANZADO_PROMPT
    | llm
    | StrOutputParser()
)

QUERIES_COMPARACION = [
    "¿Cómo afecta el cambio climático al agua y la biodiversidad de los ecosistemas de Sudamérica?",
    "¿Qué ecosistemas latinoamericanos tienen mayor potencial para energías renovables?",
]

for query in QUERIES_COMPARACION:
    print(f"\n  📋 Query: '{query}'")
    print(f"  {'─' * 65}")

    # RAG básico
    try:
        t0 = time.time()
        resp_basico = str(rag_basico.invoke(query))
        t_basico = time.time() - t0
        print(f"\n  [RAG Simple] ({t_basico:.1f}s)")
        print(f"  {resp_basico[:300].strip()}...")
    except Exception as e:
        logger.error(f"Error en RAG básico: {e}")
        raise

    # RAG avanzado (Multi-Query)
    try:
        t0 = time.time()
        resp_avanzado = str(rag_avanzado.invoke(query))
        t_avanzado = time.time() - t0
        print(f"\n  [RAG Multi-Query] ({t_avanzado:.1f}s — incluye generación de variantes)")
        print(f"  {resp_avanzado[:300].strip()}...")
    except Exception as e:
        logger.error(f"Error en RAG avanzado: {e}")
        raise

    print()

print("\n" + "═" * 70)
logger.info(
    "Lección 04 completada. "
    "Ahora tienes las herramientas para construir pipelines RAG de producción. "
    "Siguiente módulo: LangGraph — orquestar agentes que DECIDEN cómo recuperar."
)
print("═" * 70 + "\n")

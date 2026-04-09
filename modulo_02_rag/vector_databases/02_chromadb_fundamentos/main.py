"""
Módulo 02 - Lección 02: Bases de Datos Vectoriales con ChromaDB

¿Por qué esta lección importa?
─────────────────────────────
  En la Lección 01 hicimos búsqueda semántica comparando la query con
  TODOS los documentos del corpus — O(n). Con 100 docs funciona; con 1
  millón de documentos tardamos minutos.

  ChromaDB resuelve esto con un índice HNSW (Hierarchical Navigable Small
  World), una estructura de grafo que permite búsquedas aproximadas en
  tiempo O(log n). En vez de comparar con todo, navega el grafo para
  encontrar los vectores más cercanos.

  Además, ChromaDB nos da:
  ✓ Persistencia a disco (los embeddings sobreviven reinicios)
  ✓ Metadatos y filtrado (buscar solo en documentos de cierto autor)
  ✓ Integración directa con LangChain

Temas cubiertos:
  1. ChromaDB en memoria — setup, add, query
  2. Persistencia a disco — sobrevivir reinicios
  3. Metadata y filtrado — búsqueda híbrida semántica + filtros
  4. Chunking — dividir documentos grandes en fragmentos indexables
  5. Diagnóstico — inspeccionar la colección
"""

#######################
# ---- Libraries ---- #
#######################

import yaml
import shutil
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
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
        f"Configuración cargada — "
        f"colección: '{config['collection_name']}', "
        f"chunk_size: {config['chunk_size']}"
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
    logger.warning("No se encontró .env — buscando OPENAI_API_KEY en el sistema")

try:
    embeddings = OpenAIEmbeddings(model=config["embedding_model"])
    logger.info(f"Embeddings listos: {config['embedding_model']}")
except Exception as e:
    logger.error(f"No se pudo inicializar embeddings: {e}")
    raise RuntimeError("Verifica tu OPENAI_API_KEY en el archivo .env") from e


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 1: ChromaDB en Memoria
# ══════════════════════════════════════════════════════════════════════════════
#
# Modo en memoria: ideal para demos, tests y exploración rápida.
# Los datos NO persisten cuando termina el proceso.
#
# ¿Cuándo usarlo?
#   - Prototipos y demos donde no necesitas datos entre sesiones
#   - Tests unitarios donde quieres una DB limpia cada vez
#   - Exploración de datos temporales
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  BLOQUE 1: ChromaDB en Memoria")
print("═" * 70)

# Documentos sobre literatura latinoamericana
# Nota: Document tiene dos campos clave:
#   page_content → el texto que se vectorizará y recuperará
#   metadata     → diccionario de datos arbitrarios para filtrar
DOCUMENTOS_LITERATURA = [
    Document(
        page_content=(
            "Gabriel García Márquez nació en Aracataca, Colombia en 1927. "
            "Ganó el Premio Nobel de Literatura en 1982. Su obra más famosa, "
            "Cien Años de Soledad (1967), narra la historia multigeneracional "
            "de la familia Buendía en el pueblo ficticio de Macondo, mezclando "
            "realismo mágico con crítica social e histórica."
        ),
        metadata={"autor": "García Márquez", "pais": "Colombia", "genero": "realismo_magico", "año_nacimiento": 1927},
    ),
    Document(
        page_content=(
            "El amor en los tiempos del cólera (1985) de García Márquez narra "
            "el amor no correspondido de Florentino Ariza durante más de 50 años, "
            "hasta reencontrarse con Fermina Daza. La novela explora cómo el amor "
            "y el paso del tiempo se relacionan con la condición humana."
        ),
        metadata={"autor": "García Márquez", "pais": "Colombia", "genero": "realismo_magico", "año_nacimiento": 1927},
    ),
    Document(
        page_content=(
            "Jorge Luis Borges nació en Buenos Aires, Argentina en 1899. "
            "Maestro del cuento y el ensayo, exploró temas como el infinito, "
            "los laberintos, los espejos y la identidad. Sus colecciones más "
            "famosas son El Aleph (1949) y Ficciones (1944)."
        ),
        metadata={"autor": "Borges", "pais": "Argentina", "genero": "fantástico", "año_nacimiento": 1899},
    ),
    Document(
        page_content=(
            "Julio Cortázar nació en Bruselas en 1914 y vivió gran parte de su "
            "vida en Argentina y París. Rayuela (1963) es su novela más famosa: "
            "puede leerse en orden convencional o saltando entre capítulos según "
            "un tablero de instrucciones. Es un hito del boom latinoamericano."
        ),
        metadata={"autor": "Cortázar", "pais": "Argentina", "genero": "vanguardia", "año_nacimiento": 1914},
    ),
    Document(
        page_content=(
            "Isabel Allende nació en Lima, Perú en 1942, aunque es de nacionalidad "
            "chilena. La casa de los espíritus (1982) es su primera y más famosa "
            "novela, una saga familiar que mezcla historia política chilena con "
            "elementos mágicos y una fuerte perspectiva femenina."
        ),
        metadata={"autor": "Allende", "pais": "Chile", "genero": "realismo_magico", "año_nacimiento": 1942},
    ),
    Document(
        page_content=(
            "Pablo Neruda nació en Parral, Chile en 1904. Poeta prolífico, ganó "
            "el Nobel de Literatura en 1971. Sus Veinte Poemas de Amor y una "
            "Canción Desesperada (1924) es uno de los libros de poesía más "
            "vendidos en español. También escribió poesía política comprometida."
        ),
        metadata={"autor": "Neruda", "pais": "Chile", "genero": "poesía", "año_nacimiento": 1904},
    ),
]

logger.info(f"Creando colección en memoria con {len(DOCUMENTOS_LITERATURA)} documentos...")

try:
    # from_documents: crea la colección Y agrega los documentos en un solo paso.
    # Internamente: genera embeddings para cada doc y los almacena con su metadata.
    db_memoria = Chroma.from_documents(
        documents=DOCUMENTOS_LITERATURA,
        embedding=embeddings,
        collection_name="literatura_en_memoria",
    )
    logger.info("Colección en memoria creada exitosamente")
except Exception as e:
    logger.error(f"Error al crear la colección: {e}")
    raise

# Consulta básica
query_basica = "¿Quién exploró el tema del infinito y los laberintos?"
logger.info(f"Ejecutando query: '{query_basica}'")

try:
    resultados = db_memoria.similarity_search(query_basica, k=2)
    print(f"\n  Query: '{query_basica}'")
    print(f"  {'─' * 60}")
    for i, doc in enumerate(resultados, 1):
        autor = doc.metadata.get("autor", "Desconocido")
        pais  = doc.metadata.get("pais",  "?")
        print(f"  {i}. [{autor} / {pais}] {doc.page_content[:80]}...")
except Exception as e:
    logger.error(f"Error al ejecutar query: {e}")
    raise

# Consulta con scores — útil para depurar el umbral de relevancia
query_scores = "novela familiar multigeneracional latinoamericana"
logger.info(f"Query con scores: '{query_scores}'")

try:
    resultados_con_score = db_memoria.similarity_search_with_score(query_scores, k=3)
    print(f"\n  Query con scores: '{query_scores}'")
    print(f"  {'─' * 60}")
    for doc, score in resultados_con_score:
        autor = doc.metadata.get("autor", "?")
        # ChromaDB usa DISTANCIA L2 por default, no similitud coseno.
        # Menor distancia = más similar. Para similitud coseno, configurar
        # la colección con hnsw:space=cosine (ver producción avanzada).
        print(f"  [{autor}] distancia={score:.4f} | {doc.page_content[:60]}...")
except Exception as e:
    logger.error(f"Error en query con scores: {e}")
    raise

print(
    "\n  💡 ChromaDB retorna DISTANCIA (menor = mejor) por defecto."
    "\n     Para obtener similitud coseno configura la colección con"
    "\n     collection_metadata={'hnsw:space': 'cosine'}."
)


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 2: Persistencia a Disco
# ══════════════════════════════════════════════════════════════════════════════
#
# En producción, los embeddings se calculan UNA sola vez y se guardan en disco.
# Así, cuando reiniciamos el servidor no recalculamos todo.
#
# ChromaDB guarda los datos en SQLite + archivos binarios en persist_directory.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  BLOQUE 2: Persistencia a Disco")
print("═" * 70)

persist_dir = SCRIPT_DIR / config["persist_directory"]

# Limpiar una colección previa para que el demo sea reproducible
if persist_dir.exists():
    logger.warning(f"Directorio de persistencia previo encontrado en {persist_dir}. Limpiando...")
    shutil.rmtree(persist_dir)

logger.info(f"Creando colección persistente en: {persist_dir}")

try:
    db_persistente = Chroma.from_documents(
        documents=DOCUMENTOS_LITERATURA,
        embedding=embeddings,
        collection_name=config["collection_name"],
        persist_directory=str(persist_dir),
    )
    logger.info(f"Colección persistida. Documentos indexados: {db_persistente._collection.count()}")
except Exception as e:
    logger.error(f"Error al crear colección persistente: {e}")
    raise

print(f"\n  Datos guardados en: {persist_dir}")
print(f"  Documentos en la colección: {db_persistente._collection.count()}")

# Simular "reinicio del servidor" reconectando a la misma colección
logger.info("Simulando reconexión a la colección persistida...")

try:
    db_reconectada = Chroma(
        collection_name=config["collection_name"],
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )
    print(f"\n  Colección reconectada exitosamente.")
    print(f"  Documentos disponibles sin recalcular: {db_reconectada._collection.count()}")
    logger.info("Reconexión exitosa — embeddings reutilizados desde disco")
except Exception as e:
    logger.error(f"Error al reconectar a la colección: {e}")
    raise


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 3: Metadata y Filtrado — Búsqueda Híbrida
# ══════════════════════════════════════════════════════════════════════════════
#
# El verdadero poder de ChromaDB no es solo "buscar los más similares",
# sino combinar similitud semántica con FILTROS por metadatos.
#
# Casos de uso reales:
#   - "Busca documentos sobre amor, pero solo de autores colombianos"
#   - "Encuentra fragmentos técnicos, pero solo de 2023 en adelante"
#   - "Recupera capítulos similares, pero solo del volumen 2"
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  BLOQUE 3: Metadata y Filtrado")
print("═" * 70)

query_filtrada = "amor, tiempo y nostalgia"

# Sin filtro — busca en toda la colección
logger.info(f"Query sin filtro: '{query_filtrada}'")
try:
    sin_filtro = db_persistente.similarity_search(query_filtrada, k=3)
    print(f"\n  Query: '{query_filtrada}'")
    print(f"  Sin filtro (toda la colección):")
    for doc in sin_filtro:
        print(f"    - [{doc.metadata.get('autor')} / {doc.metadata.get('pais')}] {doc.page_content[:60]}...")
except Exception as e:
    logger.error(f"Error en query sin filtro: {e}")
    raise

# Con filtro por país — solo autores argentinos
logger.info("Ejecutando query con filtro por país: Argentina")
try:
    con_filtro_pais = db_persistente.similarity_search(
        query_filtrada,
        k=3,
        filter={"pais": "Argentina"},
    )
    print(f"\n  Con filtro (pais='Argentina'):")
    for doc in con_filtro_pais:
        print(f"    - [{doc.metadata.get('autor')} / {doc.metadata.get('pais')}] {doc.page_content[:60]}...")
except Exception as e:
    logger.error(f"Error en query con filtro: {e}")
    raise

# Con filtro por género literario
logger.info("Ejecutando query con filtro por género: realismo_magico")
try:
    con_filtro_genero = db_persistente.similarity_search(
        "familia y historia",
        k=5,
        filter={"genero": "realismo_magico"},
    )
    print(f"\n  Query 'familia y historia' con filtro (genero='realismo_magico'):")
    for doc in con_filtro_genero:
        print(f"    - [{doc.metadata.get('autor')}] {doc.page_content[:65]}...")
except Exception as e:
    logger.error(f"Error en query con filtro de género: {e}")
    raise

print(
    "\n  💡 El filtrado por metadata es ANTES de la búsqueda semántica."
    "\n     ChromaDB primero filtra los documentos que cumplen el where, "
    "\n     luego busca el más similar entre esos. Esto hace que k resultados"
    "\n     sean k del subconjunto filtrado, no de toda la colección."
)


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 4: Chunking — Dividir documentos para indexarlos bien
# ══════════════════════════════════════════════════════════════════════════════
#
# Los modelos de embeddings tienen un límite de tokens (típicamente 8192 para
# text-embedding-3-small). Documentos más largos se truncan, perdiendo info.
#
# Más importante: embeddings de textos MUY largos capturan un "promedio"
# de muchos temas, lo que diluye la señal semántica. Un chunk de 500 tokens
# sobre un tema específico da mucho mejor retrieval que un capítulo entero.
#
# Parámetros clave del RecursiveCharacterTextSplitter:
#   chunk_size:    Tamaño objetivo de cada fragmento (en caracteres, no tokens)
#   chunk_overlap: Cuántos caracteres se solapan entre chunks consecutivos.
#                  El overlap evita perder contexto en los bordes de los chunks.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  BLOQUE 4: Chunking — Dividir documentos grandes")
print("═" * 70)

TEXTO_LARGO = """
La novela latinoamericana del siglo XX representa uno de los fenómenos literarios
más importantes de la historia universal. El llamado "Boom Latinoamericano" de los
años 60 y 70 catapultó a autores como García Márquez, Cortázar, Vargas Llosa y
Carlos Fuentes a la escena internacional.

El realismo mágico, corriente literaria donde eventos sobrenaturales se integran
naturalmenteen la narrativa, es el sello del Boom. No se presenta como algo
extraordinario: los muertos hablan, los años duran cien, las mariposas amarillas
anuncian la llegada de un personaje. Para los personajes y el narrador, esto es
simplemente la realidad.

García Márquez desarrolló el realismo mágico en sus novelas basado en la forma
en que su abuela le contaba historias de niño: mezcla de lo real y lo fantástico
como si fueran lo mismo. Esta técnica narrativa captura algo profundo sobre la
experiencia latinoamericana: la coexistencia de modernidad y tradición oral, de
racionalismo europeo y cosmovisión indígena.

Cortázar, en cambio, exploró lo fantástico como ruptura de la normalidad. En sus
cuentos, un hombre sueña que es una mariposa o viceversa; una autopista se
convierte en una trampa; un paciente en el quirófano es simultáneamente un
guerrero azteca en un sacrificio. Cortázar subvierte la lógica causal.

El Boom no fue solo un fenómeno estético. Sus autores eran comprometidos
políticamente: apoyaron la Revolución Cubana inicialmente (aunque muchos
rompieron con ella después), denunciaron dictaduras y reflexionaron sobre la
identidad latinoamericana postcolonial.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=config["chunk_size"],
    chunk_overlap=config["chunk_overlap"],
    # RecursiveCharacterTextSplitter intenta dividir en este orden de separadores:
    # primero por párrafos (\n\n), luego por líneas (\n), luego por oraciones (. ! ?),
    # y finalmente por palabras o caracteres. Así preserva la coherencia del texto.
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
)

chunks = splitter.split_text(TEXTO_LARGO)

logger.info(
    f"Texto dividido en {len(chunks)} chunks "
    f"(chunk_size={config['chunk_size']}, overlap={config['chunk_overlap']})"
)

print(f"\n  Texto original: {len(TEXTO_LARGO):,} caracteres")
print(f"  Número de chunks: {len(chunks)}")
print(f"  chunk_size={config['chunk_size']}, overlap={config['chunk_overlap']}\n")

for i, chunk in enumerate(chunks):
    print(f"  Chunk {i+1} ({len(chunk)} chars):")
    print(f"  │ {chunk[:100].strip()}...")
    print()

print(
    "  💡 Nota el overlap: los últimos caracteres del Chunk N aparecen"
    "\n     también al inicio del Chunk N+1. Esto garantiza que una frase"
    "\n     que cae en el borde entre dos chunks no se pierde."
)

# Limpiar datos de persistencia al finalizar el demo
logger.info(f"Limpiando directorio de persistencia del demo: {persist_dir}")
if persist_dir.exists():
    shutil.rmtree(persist_dir)
    logger.info("Directorio limpiado")

print("\n" + "═" * 70)
logger.info(
    "Lección 02 completada. "
    "Siguiente: Lección 03 — construir un pipeline RAG completo."
)
print("═" * 70 + "\n")

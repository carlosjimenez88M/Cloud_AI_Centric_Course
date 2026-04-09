"""
Módulo 02 - Lección 01: Embeddings y Similitud Semántica

¿Por qué esta lección importa?
─────────────────────────────
  Los embeddings son la unidad fundamental de todo sistema RAG, motor de
  búsqueda semántica y clasificador de texto moderno. Antes de meter
  documentos en una base de datos vectorial, necesitamos entender QUÉ son
  los vectores que almacenamos y QUÉ mide la similitud entre ellos.

  Sin esta base, ChromaDB y RAG son cajas negras. Con ella, puedes
  diagnosticar por qué tu sistema recupera documentos irrelevantes.

Temas cubiertos:
  1. ¿Qué es un embedding? (de texto a puntos en el espacio)
  2. Generación de embeddings con OpenAI (text-embedding-3-small)
  3. Similitud del coseno — implementación manual paso a paso
  4. Por qué coseno es mejor que distancia euclidiana para texto
  5. Búsqueda semántica básica sin base de datos vectorial
"""

#######################
# ---- Libraries ---- #
#######################

import yaml
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# ── Logger del módulo compartido ──────────────────────────────────────────────
# El paquete modulo_02_vector_databases está instalado en el workspace de uv
# (pyproject.toml con src-layout). Se importa exactamente igual que cualquier
# dependencia de terceros — sin manipular sys.path manualmente.
from modulo_02_vector_databases.shared.logger import get_logger

SCRIPT_DIR = Path(__file__).parent

###########################
# ---- Logger Design ---- #
###########################

# Convencion: pasar __name__ para que el log muestre qué archivo lo emite.
# Así puedes saber exactamente de dónde viene cada mensaje.
logger = get_logger(__name__)

##########################
# ---- Load Config ---- #
##########################

try:
    with open(SCRIPT_DIR / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.info(f"Configuración cargada — modelo de embeddings: {config['embedding_model']}")
except FileNotFoundError:
    logger.error("No se encontró config.yaml en el directorio de la lección")
    raise
except yaml.YAMLError as e:
    logger.error(f"Error al parsear config.yaml: {e}")
    raise

######################
# ---- Call API ---- #
######################

# load_dotenv() busca el archivo .env en el directorio actual y sus padres.
# Retorna True si encontró el archivo, False si no. No lanza excepción.
if not load_dotenv():
    logger.warning("No se encontró .env — buscando OPENAI_API_KEY en variables de entorno del sistema")

try:
    embeddings = OpenAIEmbeddings(model=config["embedding_model"])
    logger.info(f"Modelo de embeddings inicializado: {config['embedding_model']}")
except Exception as e:
    logger.error(f"No se pudo inicializar OpenAIEmbeddings: {e}")
    raise RuntimeError(
        "Verifica que OPENAI_API_KEY esté configurada en tu archivo .env"
    ) from e


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 1: ¿Qué es un embedding?
# ══════════════════════════════════════════════════════════════════════════════
#
# Un embedding es una lista de números (vector) que representa el SIGNIFICADO
# de un texto en un espacio matemático de alta dimensión.
#
# La idea clave: textos con significados similares producen vectores que
# "apuntan en la misma dirección" en ese espacio. Textos diferentes apuntan
# en direcciones distintas.
#
# Analogía geográfica:
#   Si mapeamos ciudades en un plano 2D por latitud/longitud, ciudades cercanas
#   tienen cosas en común (clima, idioma). Los embeddings hacen lo mismo para
#   el SIGNIFICADO: "gato" y "felino" quedan cerca; "gato" y "democracia" lejos.
#
# text-embedding-3-small de OpenAI produce vectores de 1536 dimensiones.
# Cada dimensión captura algún aspecto semántico del texto.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  BLOQUE 1: ¿Qué es un embedding?")
print("═" * 70)

frase_ejemplo = "El coronel no tenía quien le escribiera"

logger.info(f"Generando embedding para: '{frase_ejemplo}'")

try:
    vector = embeddings.embed_query(frase_ejemplo)
except Exception as e:
    logger.error(f"Fallo al llamar a la API de OpenAI: {e}")
    raise

vector_np = np.array(vector)

print(f"\n  Frase:              '{frase_ejemplo}'")
print(f"  Dimensiones:         {len(vector):,}  ← 1536 números representan el significado")
print(f"  Primeras 8 dims:     {vector_np[:8].round(5)}")
print(f"  Norma del vector:    {np.linalg.norm(vector_np):.6f}  ← OpenAI normaliza a ~1.0")
print(
    "\n  💡 Observa que la norma es casi exactamente 1.0. OpenAI entrega vectores"
    "\n     unitarios (normalizados). Esto tiene una implicación importante que veremos"
    "\n     en el Bloque 3."
)


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 2: Similitud del coseno — implementación desde cero
# ══════════════════════════════════════════════════════════════════════════════
#
# La similitud del coseno mide el ángulo θ entre dos vectores, NO su distancia.
#
#   cos(θ) = (A · B) / (||A|| × ||B||)
#
# Donde:
#   A · B   → producto punto (dot product): suma de A[i] × B[i] para todo i
#   ||A||   → norma euclidiana: √(A[0]² + A[1]² + ... + A[n]²)
#
# Rango de valores:
#    1.0  → vectores idénticos (el mismo texto exacto)
#   ~0.9  → textos muy similares
#   ~0.7  → textos relacionados
#   ~0.5  → textos del mismo dominio
#    0.0  → textos completamente sin relación
#   -1.0  → vectores opuestos (raro en embeddings de texto moderno)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  BLOQUE 2: Similitud del coseno desde cero")
print("═" * 70)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Calcula la similitud del coseno entre dos vectores de embeddings.

    Implementamos esto manualmente para entender QUÉ hace ChromaDB y las
    bases de datos vectoriales cuando comparan documentos. En producción,
    el cálculo lo hace el índice HNSW de la base de datos, no nosotros.

    Args:
        vec_a: Vector de embedding A (numpy array 1D)
        vec_b: Vector de embedding B (numpy array 1D)

    Returns:
        float entre -1.0 y 1.0. Mayor = más similar.

    Raises:
        ValueError: Si los vectores no son 1D, tienen distinta longitud,
                    o alguno tiene norma cero (texto vacío o inválido).

    Example:
        >>> a = np.array([1.0, 0.0, 0.0])
        >>> b = np.array([1.0, 0.0, 0.0])
        >>> cosine_similarity(a, b)
        1.0
    """
    if vec_a.ndim != 1 or vec_b.ndim != 1:
        raise ValueError(
            f"Se esperan vectores 1D. Recibidos: {vec_a.shape} y {vec_b.shape}. "
            f"¿Pasaste una matriz en lugar de un vector?"
        )
    if vec_a.shape[0] != vec_b.shape[0]:
        raise ValueError(
            f"Los vectores deben tener la misma longitud: "
            f"{vec_a.shape[0]} ≠ {vec_b.shape[0]}. "
            f"¿Estás mezclando modelos de embeddings diferentes?"
        )

    norma_a = np.linalg.norm(vec_a)
    norma_b = np.linalg.norm(vec_b)

    if norma_a == 0.0:
        raise ValueError("vec_a tiene norma 0. ¿El texto estaba vacío?")
    if norma_b == 0.0:
        raise ValueError("vec_b tiene norma 0. ¿El texto estaba vacío?")

    return float(np.dot(vec_a, vec_b) / (norma_a * norma_b))


# Corpus de frases de tres dominios distintos para demostrar clustering semántico
FRASES = {
    # ── Literatura latinoamericana ─────────────────────────────────────────
    "gabo_soledad": (
        "Muchos años después, frente al pelotón de fusilamiento, el coronel Aureliano "
        "Buendía habría de recordar aquella tarde remota en que su padre lo llevó a "
        "conocer el hielo"
    ),
    "gabo_amor": (
        "Era inevitable: el olor de las almendras amargas le recordaba siempre el "
        "destino de los amores contrariados"
    ),
    "borges_biblioteca": (
        "El universo, que otros llaman la Biblioteca, se compone de un número indefinido, "
        "y tal vez infinito, de galerías hexagonales"
    ),
    "cortazar_rayuela": (
        "Andábamos sin buscarnos pero sabiendo que andábamos para encontrarnos"
    ),
    # ── Inteligencia Artificial ────────────────────────────────────────────
    "ia_llm": (
        "Los modelos de lenguaje de gran escala aprenden distribuciones estadísticas "
        "sobre tokens de texto para predecir la siguiente palabra dado un contexto"
    ),
    "ia_embeddings": (
        "Los embeddings son representaciones vectoriales densas de texto que capturan "
        "relaciones semánticas y sintácticas en un espacio de alta dimensión"
    ),
    "ia_rag": (
        "La generación aumentada por recuperación combina un retriever de documentos "
        "con un modelo generativo para producir respuestas fundamentadas en hechos"
    ),
    # ── Gastronomía latinoamericana ────────────────────────────────────────
    "ceviche": (
        "El ceviche peruano se prepara con pescado fresco marinado en jugo de limón, "
        "ají limo, cebolla morada y cilantro fresco al momento de servir"
    ),
    "asado": (
        "El asado argentino es una tradición cultural donde la carne vacuna se cocina "
        "lentamente sobre brasas de quebracho colorado en una parrilla"
    ),
}

logger.info(f"Generando embeddings para {len(FRASES)} frases del corpus...")

try:
    textos   = list(FRASES.values())
    claves   = list(FRASES.keys())
    vectores_lista = embeddings.embed_documents(textos)
    vectores = {claves[i]: np.array(vectores_lista[i]) for i in range(len(claves))}
    logger.info("Embeddings del corpus generados correctamente")
except Exception as e:
    logger.error(f"Error al generar embeddings del corpus: {e}")
    raise

print("\n  Comparaciones de similitud del coseno")
print("  (1.0 = idéntico, 0.0 = sin relación)\n")

comparaciones = [
    # Dentro del mismo dominio → deberían tener ALTA similitud
    ("gabo_soledad",     "gabo_amor",        "Gabo vs Gabo               "),
    ("gabo_soledad",     "cortazar_rayuela",  "García Márquez vs Cortázar "),
    ("ia_llm",           "ia_embeddings",    "LLM vs Embeddings (IA)     "),
    ("ia_llm",           "ia_rag",           "LLM vs RAG (IA)            "),
    ("ceviche",          "asado",            "Ceviche vs Asado           "),
    # Entre dominios → deberían tener BAJA similitud
    ("gabo_soledad",     "ia_llm",           "Literatura vs IA           "),
    ("borges_biblioteca","ceviche",          "Borges vs Gastronomía      "),
    ("ia_rag",           "asado",            "RAG vs Asado               "),
]

for key_a, key_b, etiqueta in comparaciones:
    try:
        sim   = cosine_similarity(vectores[key_a], vectores[key_b])
        barra = "█" * int(sim * 25) + "░" * (25 - int(sim * 25))
        print(f"  {etiqueta} [{barra}] {sim:.4f}")
    except ValueError as e:
        logger.warning(f"No se pudo comparar {key_a} vs {key_b}: {e}")

print(
    "\n  💡 Observa cómo frases del mismo dominio tienen alta similitud,"
    "\n     aunque no compartan palabras exactas. Eso es captura de SIGNIFICADO,"
    "\n     no comparación de palabras clave (como haría un buscador clásico)."
)


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 3: Coseno vs Distancia Euclidiana — ¿Por qué importa la diferencia?
# ══════════════════════════════════════════════════════════════════════════════
#
# La distancia euclidiana mide cuánto espacio hay entre dos puntos.
# Problema para texto: un párrafo largo tiene un vector con mayor magnitud que
# una frase corta sobre el mismo tema. La distancia euclidiana los verá lejos
# aunque hablen de lo mismo.
#
# La similitud del coseno solo mide el ÁNGULO. Ignora la magnitud.
# Por eso, "la IA cambia el mundo" y "la inteligencia artificial está
# transformando profundamente todos los sectores de la sociedad moderna"
# quedan cerca con coseno, aunque uno sea más largo.
#
# IMPORTANTE: OpenAI normaliza sus vectores (norma ≈ 1.0).
# Cuando ambos vectores son unitarios, cos(θ) y distancia euclidiana son
# equivalentes matemáticamente. Pero modelos como sentence-transformers
# NO normalizan, y ahí la diferencia es crítica.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  BLOQUE 3: Coseno vs Distancia Euclidiana")
print("═" * 70)

frase_corta   = "La inteligencia artificial cambia el mundo"
frase_larga   = (
    "La inteligencia artificial está transformando profundamente todos los sectores "
    "de la sociedad moderna: la medicina, la educación, la industria, el entretenimiento "
    "y la forma en que los seres humanos se relacionan entre sí y con las máquinas, "
    "redefiniendo el mercado laboral y los paradigmas de productividad globales"
)
frase_ajena   = "El dulce de leche es un producto lácteo típico del Río de la Plata"

logger.info("Generando embeddings para la comparación coseno vs euclidiana...")

try:
    v_corta  = np.array(embeddings.embed_query(frase_corta))
    v_larga  = np.array(embeddings.embed_query(frase_larga))
    v_ajena  = np.array(embeddings.embed_query(frase_ajena))
except Exception as e:
    logger.error(f"Error al generar embeddings: {e}")
    raise


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distancia euclidiana estándar. Mayor = más lejanos."""
    return float(np.linalg.norm(a - b))


print(f"\n  Frase corta:    '{frase_corta}'")
print(f"  Frase larga:    '{frase_larga[:65]}...'")
print(f"  Frase ajena:    '{frase_ajena}'")
print(f"\n  Norma frase corta: {np.linalg.norm(v_corta):.6f}")
print(f"  Norma frase larga: {np.linalg.norm(v_larga):.6f}")

print("\n  ── Corta vs Larga (MISMO TEMA) ──")
cos_cl = cosine_similarity(v_corta, v_larga)
euc_cl = euclidean_distance(v_corta, v_larga)
print(f"  Similitud coseno:     {cos_cl:.4f}  ← Alta (mismo significado)")
print(f"  Distancia euclidiana: {euc_cl:.4f}")

print("\n  ── Corta vs Ajena (TEMAS DISTINTOS) ──")
cos_ca = cosine_similarity(v_corta, v_ajena)
euc_ca = euclidean_distance(v_corta, v_ajena)
print(f"  Similitud coseno:     {cos_ca:.4f}  ← Baja (temas distintos)")
print(f"  Distancia euclidiana: {euc_ca:.4f}")

print(
    "\n  💡 Con OpenAI los vectores son unitarios, así que las métricas son"
    "\n     equivalentes. Pero en sentence-transformers o modelos sin normalizar,"
    "\n     siempre usa coseno. Es la métrica correcta para comparar significado."
)


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 4: Búsqueda semántica básica (sin base de datos)
# ══════════════════════════════════════════════════════════════════════════════
#
# Podemos hacer búsqueda semántica sin ChromaDB ni ninguna DB:
# solo calcular la similitud entre la query y cada documento,
# luego ordenar de mayor a menor.
#
# Limitación: escala O(n) — tenemos que comparar con TODOS los documentos.
# Con 100 docs es trivial. Con 1 millón, es inviable.
# → Por eso existen las bases de datos vectoriales con índices HNSW.
# Las veremos en la Lección 02.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  BLOQUE 4: Búsqueda Semántica Básica (sin base de datos)")
print("═" * 70)

CORPUS_BUSQUEDA = [
    "García Márquez nació en Aracataca, Colombia, y ganó el Nobel de Literatura en 1982 por Cien Años de Soledad",
    "Borges exploró laberintos, espejos y el infinito en El Aleph, Ficciones y otras colecciones de cuentos",
    "Cortázar inventó la rayuela literaria: una novela donde el lector elige su propio orden de capítulos",
    "Isabel Allende narra sagas familiares y feminismo en contextos históricos latinoamericanos",
    "Pablo Neruda escribió odas, elegías y poemas de amor, incluyendo los célebres Veinte Poemas de Amor",
    "Los transformers y la atención multi-cabeza revolucionaron el procesamiento del lenguaje natural en 2017",
    "ChromaDB es una base de datos vectorial open-source optimizada para aplicaciones de IA generativa",
    "El fine-tuning adapta un modelo preentrenado a una tarea específica con pocos datos adicionales",
    "FAISS de Meta permite búsqueda eficiente en espacios vectoriales de alta dimensión a escala de millones",
    "El ceviche se tigre es el líquido sobrante del marinado, considerado un elixir en la cocina peruana",
    "La chicha morada es una bebida no fermentada hecha de maíz morado, frutas y especias del Perú",
    "El mate cimarrón se bebe amargo en Uruguay y Argentina como ritual cotidiano de comunidad",
]


def buscar_semanticamente(
    query: str,
    corpus: list[str],
    modelo: OpenAIEmbeddings,
    top_k: int = 3,
) -> list[tuple[str, float]]:
    """
    Búsqueda semántica sobre un corpus usando similitud del coseno.

    Esta función es el núcleo de lo que hace ChromaDB, pero sin los beneficios
    de indexación. Úsala para entender el mecanismo, no para producción.

    Args:
        query:   El texto de búsqueda (pregunta o frase)
        corpus:  Lista de textos donde buscar
        modelo:  Modelo de embeddings para vectorizar query y corpus
        top_k:   Cuántos resultados retornar

    Returns:
        Lista de (texto, similitud) ordenada de mayor a menor similitud.

    Raises:
        ValueError: Si el corpus está vacío
        RuntimeError: Si falla la llamada a la API de embeddings
    """
    if not corpus:
        raise ValueError("El corpus no puede estar vacío")

    if top_k > len(corpus):
        logger.warning(
            f"top_k={top_k} excede el tamaño del corpus ({len(corpus)}). "
            f"Ajustando a {len(corpus)}"
        )
        top_k = len(corpus)

    try:
        logger.debug(f"Vectorizando query: '{query[:55]}...'")
        v_query  = np.array(modelo.embed_query(query))

        logger.debug(f"Vectorizando {len(corpus)} documentos del corpus...")
        v_corpus = np.array(modelo.embed_documents(corpus))

    except Exception as e:
        logger.error(f"Fallo al vectorizar: {e}")
        raise RuntimeError(f"Error en la búsqueda semántica: {e}") from e

    similitudes = [
        (corpus[i], cosine_similarity(v_query, v_corpus[i]))
        for i in range(len(corpus))
    ]
    similitudes.sort(key=lambda x: x[1], reverse=True)

    logger.info(
        f"Búsqueda completada — "
        f"mejor resultado: {similitudes[0][1]:.4f}"
    )
    return similitudes[:top_k]


CONSULTAS = [
    "¿Quién escribió sobre laberintos y el infinito?",
    "¿Cómo funcionan las bases de datos vectoriales para IA?",
    "¿Cuáles son las bebidas típicas de América del Sur?",
]

for consulta in CONSULTAS:
    print(f"\n  🔍 Query: '{consulta}'")
    print(f"  {'─' * 65}")
    try:
        resultados = buscar_semanticamente(consulta, CORPUS_BUSQUEDA, embeddings, top_k=3)
        for i, (texto, sim) in enumerate(resultados, 1):
            print(f"  {i}. [{sim:.4f}] {texto[:75]}...")
    except (ValueError, RuntimeError) as e:
        logger.error(f"Error en la búsqueda: {e}")

print("\n" + "═" * 70)
logger.info(
    "Lección 01 completada. "
    "Siguiente: ChromaDB — escalar la búsqueda semántica a millones de documentos."
)
print("═" * 70 + "\n")

"""
Módulo 02 - Lección 02: Exercise — Knowledge Base de Autores Latinoamericanos

¿Qué construimos aquí?
─────────────────────
  Un sistema de consulta semántica sobre una base de conocimiento de
  autores latinoamericanos del siglo XX. A diferencia del demo, aquí
  construimos algo con propósito real:

    1. Indexamos información biográfica y literaria de 7 autores
    2. Demostramos filtrado combinado (semántico + metadata)
    3. Implementamos un "asistente de consulta" básico

  Este ejercicio simula lo que haría un sistema RAG en producción,
  pero sin el componente de generación (LLM). Solo retrieval.
"""

#######################
# ---- Libraries ---- #
#######################

from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from modulo_02_vector_databases.shared.logger import get_logger

###########################
# ---- Logger Design ---- #
###########################

logger = get_logger(__name__)

######################
# ---- Call API ---- #
######################

if not load_dotenv():
    logger.warning("No se encontró .env — buscando OPENAI_API_KEY en el sistema")

try:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    logger.info("Modelo de embeddings inicializado")
except Exception as e:
    logger.error(f"No se pudo inicializar embeddings: {e}")
    raise RuntimeError("Verifica tu OPENAI_API_KEY") from e


# ══════════════════════════════════════════════════════════════════════════════
# Base de Conocimiento — Documentos a indexar
# ══════════════════════════════════════════════════════════════════════════════
#
# Cada documento tiene:
#   page_content → texto que se vectorizará
#   metadata     → diccionario con campos para filtrado
#
# Diseño de metadata: piensa en qué filtros necesitarás antes de crear la DB.
# Una vez indexados los docs, no puedes agregar campos de metadata sin
# re-indexar. Planifica metadata como si fuera el esquema de una tabla SQL.
# ══════════════════════════════════════════════════════════════════════════════

KNOWLEDGE_BASE = [
    # ── García Márquez ────────────────────────────────────────────────────────
    Document(
        page_content=(
            "Cien Años de Soledad (1967) narra siete generaciones de la familia Buendía "
            "en el pueblo ficticio de Macondo. La novela explora los ciclos de la historia, "
            "la soledad inherente a la condición humana y el inexorable paso del tiempo. "
            "El realismo mágico está presente en cada página: alfombras voladoras, "
            "lluvia de flores amarillas, un fantasma que ronda la casa. Considerada "
            "una de las mejores novelas del siglo XX y pieza central del Boom Latinoamericano."
        ),
        metadata={"autor": "García Márquez", "pais": "Colombia", "obra": "Cien Años de Soledad", "año": 1967, "temas": "soledad,familia,tiempo,realismo_magico"},
    ),
    Document(
        page_content=(
            "El coronel no tiene quien le escriba (1961) es una novela corta de García Márquez. "
            "Un coronel veterano de la guerra civil colombiana espera durante 15 años una "
            "pensión del gobierno que nunca llega. La historia es una alegoría de la esperanza "
            "obstinada frente a la burocracia, la pobreza y el tiempo que pasa sin cambio. "
            "Considerada por el propio autor su obra más lograda estilísticamente."
        ),
        metadata={"autor": "García Márquez", "pais": "Colombia", "obra": "El coronel no tiene quien le escriba", "año": 1961, "temas": "espera,dignidad,pobreza,politica"},
    ),
    # ── Borges ───────────────────────────────────────────────────────────────
    Document(
        page_content=(
            "El Aleph (1949) es un cuento de Jorge Luis Borges donde el narrador descubre "
            "un punto en el espacio que contiene todos los puntos: el Aleph, desde donde "
            "se puede ver simultáneamente todo el universo. Borges usa este concepto para "
            "explorar el infinito, la memoria total y los límites del lenguaje para describir "
            "lo inefable. El cuento es una meditación sobre la imposibilidad de capturar "
            "la totalidad de la experiencia en palabras."
        ),
        metadata={"autor": "Borges", "pais": "Argentina", "obra": "El Aleph", "año": 1949, "temas": "infinito,memoria,universo,lenguaje"},
    ),
    Document(
        page_content=(
            "La Biblioteca de Babel (1941) imagina un universo compuesto enteramente de "
            "una biblioteca infinita con todas las combinaciones posibles de letras. "
            "Algunos libros contienen verdades absolutas; la mayoría, sinsentido puro. "
            "Borges usa la metáfora para reflexionar sobre el conocimiento, la búsqueda "
            "de sentido en un universo caótico, y la condición del ser humano perdido "
            "en un sistema que lo supera. Es proto-digital: anticipa internet."
        ),
        metadata={"autor": "Borges", "pais": "Argentina", "obra": "La Biblioteca de Babel", "año": 1941, "temas": "infinito,conocimiento,caos,biblioteca"},
    ),
    # ── Cortázar ─────────────────────────────────────────────────────────────
    Document(
        page_content=(
            "Rayuela (1963) de Julio Cortázar es una anti-novela o novela abierta. "
            "Puede leerse de dos formas: en orden convencional del 1 al 56, o siguiendo "
            "un tablero de instrucciones que lleva al lector a saltar entre capítulos. "
            "La historia sigue a Horacio Oliveira entre París y Buenos Aires, su relación "
            "con la Maga, y su búsqueda existencial de un 'kibbutz del deseo', un centro "
            "espiritual auténtico fuera de la convención burguesa."
        ),
        metadata={"autor": "Cortázar", "pais": "Argentina", "obra": "Rayuela", "año": 1963, "temas": "existencialismo,amor,libertad,vanguardia"},
    ),
    Document(
        page_content=(
            "Casa Tomada (1946), cuento de Cortázar, narra a dos hermanos cuya casa "
            "es gradualmente invadida por una presencia innombrada. Sin describir qué es "
            "la amenaza, Cortázar crea terror puro: lo fantástico emerge de lo cotidiano. "
            "El cuento es interpretado como alegoría del peronismo (la presencia que desplaza "
            "a la clase media argentina de sus espacios), aunque Cortázar negó esta lectura."
        ),
        metadata={"autor": "Cortázar", "pais": "Argentina", "obra": "Casa Tomada", "año": 1946, "temas": "fantástico,miedo,alienación,politica"},
    ),
    # ── Vargas Llosa ─────────────────────────────────────────────────────────
    Document(
        page_content=(
            "La ciudad y los perros (1963) de Mario Vargas Llosa transcurre en el Colegio "
            "Militar Leoncio Prado de Lima. La novela denuncia la violencia, el machismo "
            "y la corrupción institucional mediante la historia de un grupo de cadetes. "
            "Fue quemada públicamente en Lima por el ejército peruano, lo que paradójicamente "
            "la convirtió en un best-seller internacional. Vargas Llosa ganó el Nobel en 2010."
        ),
        metadata={"autor": "Vargas Llosa", "pais": "Perú", "obra": "La ciudad y los perros", "año": 1963, "temas": "violencia,militarismo,masculinidad,denuncia"},
    ),
    # ── Rulfo ────────────────────────────────────────────────────────────────
    Document(
        page_content=(
            "Pedro Páramo (1955) de Juan Rulfo es quizás la novela más influyente de la "
            "literatura latinoamericana. Juan Preciado viaja a Comala buscando a su padre, "
            "Pedro Páramo, solo para descubrir que el pueblo está habitado por muertos. "
            "La narración es fragmentada, los tiempos se mezclan, los vivos y los muertos "
            "hablan con la misma voz. García Márquez dijo que después de leerla, tuvo que "
            "aprender a escribir de nuevo."
        ),
        metadata={"autor": "Rulfo", "pais": "México", "obra": "Pedro Páramo", "año": 1955, "temas": "muerte,identidad,poder,fragmentación"},
    ),
    # ── Allende ──────────────────────────────────────────────────────────────
    Document(
        page_content=(
            "La casa de los espíritus (1982) de Isabel Allende narra cuatro generaciones "
            "de la familia Trueba en Chile, desde inicios del siglo XX hasta el golpe de "
            "estado de 1973. La historia alterna entre el patriarca conservador Esteban "
            "Trueba y su esposa clarividente Clara, sus hijos y nieta Blanca y Alba. "
            "Es considerada un hito del realismo mágico con perspectiva feminista."
        ),
        metadata={"autor": "Allende", "pais": "Chile", "obra": "La casa de los espíritus", "año": 1982, "temas": "familia,feminismo,historia,politica,realismo_magico"},
    ),
]


def construir_knowledge_base(
    documentos: list[Document],
    modelo_embeddings: OpenAIEmbeddings,
    nombre_coleccion: str,
) -> Chroma:
    """
    Construye e indexa la base de conocimiento en ChromaDB (en memoria).

    Args:
        documentos:          Lista de Document a indexar
        modelo_embeddings:   Modelo para vectorizar los textos
        nombre_coleccion:    Nombre de la colección en ChromaDB

    Returns:
        Instancia de Chroma con todos los documentos indexados

    Raises:
        ValueError: Si la lista de documentos está vacía
        RuntimeError: Si falla la creación de la colección
    """
    if not documentos:
        raise ValueError("No se pueden indexar 0 documentos. Proporciona al menos uno.")

    logger.info(f"Indexando {len(documentos)} documentos en colección '{nombre_coleccion}'...")

    try:
        db = Chroma.from_documents(
            documents=documentos,
            embedding=modelo_embeddings,
            collection_name=nombre_coleccion,
        )
        count = db._collection.count()
        logger.info(f"Knowledge base lista: {count} documentos indexados")
        return db
    except Exception as e:
        logger.error(f"Error al construir la knowledge base: {e}")
        raise RuntimeError(f"Fallo en la indexación: {e}") from e


def consultar(
    db: Chroma,
    pregunta: str,
    k: int = 3,
    filtro: dict | None = None,
) -> list[Document]:
    """
    Consulta la knowledge base y retorna los documentos más relevantes.

    Args:
        db:       Instancia de Chroma a consultar
        pregunta: Texto de búsqueda
        k:        Número de resultados
        filtro:   Dict de metadata para filtrar (opcional)

    Returns:
        Lista de Document ordenados por relevancia

    Raises:
        RuntimeError: Si falla la consulta a ChromaDB
    """
    try:
        kwargs: dict = {"k": k}
        if filtro:
            kwargs["filter"] = filtro
            logger.debug(f"Query con filtro {filtro}: '{pregunta[:50]}'")
        else:
            logger.debug(f"Query sin filtro: '{pregunta[:50]}'")

        return db.similarity_search(pregunta, **kwargs)
    except Exception as e:
        logger.error(f"Error al consultar ChromaDB: {e}")
        raise RuntimeError(f"Fallo en la consulta: {e}") from e


def imprimir_resultados(pregunta: str, resultados: list[Document]) -> None:
    """Imprime resultados de búsqueda de forma legible."""
    print(f"\n  🔍 '{pregunta}'")
    print(f"  {'─' * 65}")
    if not resultados:
        print("  ⚠️  Sin resultados para esta consulta y filtros.")
        return
    for i, doc in enumerate(resultados, 1):
        autor = doc.metadata.get("autor", "?")
        obra  = doc.metadata.get("obra",  "?")
        año   = doc.metadata.get("año",   "?")
        print(f"  {i}. [{autor} – {obra} ({año})]")
        print(f"     {doc.page_content[:110]}...")
        print()


# ── Main execution ────────────────────────────────────────────────────────────

print("\n" + "═" * 70)
print("  EXERCISE: Knowledge Base de Literatura Latinoamericana")
print("═" * 70)

# 1. Construir la knowledge base
kb = construir_knowledge_base(KNOWLEDGE_BASE, embeddings, "kb_literatura")

# 2. Consultas de demostración

# Consulta 1: Búsqueda semántica pura
imprimir_resultados(
    "¿Quién exploró el infinito y los laberintos matemáticos?",
    consultar(kb, "¿Quién exploró el infinito y los laberintos matemáticos?", k=2),
)

# Consulta 2: Búsqueda con filtro por país
imprimir_resultados(
    "obras sobre política y denuncia social",
    consultar(
        kb,
        "obras sobre política y denuncia social",
        k=3,
        filtro={"pais": "Argentina"},
    ),
)

# Consulta 3: Búsqueda sobre realismo mágico
imprimir_resultados(
    "¿En qué novelas se mezclan lo real y lo sobrenatural?",
    consultar(kb, "¿En qué novelas se mezclan lo real y lo sobrenatural?", k=3),
)

# Consulta 4: Búsqueda por autor específico
imprimir_resultados(
    "la muerte y el tiempo en la narrativa",
    consultar(
        kb,
        "la muerte y el tiempo en la narrativa",
        k=2,
        filtro={"autor": "Rulfo"},
    ),
)

# Consulta 5: Búsqueda temporal (obras antes de 1960)
imprimir_resultados(
    "obras experimentales que rompen con la narrativa lineal",
    consultar(
        kb,
        "obras experimentales que rompen con la narrativa lineal",
        k=3,
        filtro={"año": {"$lt": 1960}},
    ),
)

print("\n" + "═" * 70)
logger.info(
    "Exercise 02 completado. "
    "Este retriever es el primer componente de un pipeline RAG. "
    "En la Lección 03 agregaremos el generador (LLM) para completarlo."
)
print("═" * 70 + "\n")

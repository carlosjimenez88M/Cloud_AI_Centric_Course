"""
Módulo 02 — Bases de Datos Vectoriales, Embeddings y RAG
Entry point principal del módulo

Uso:
    python modulo_02_vector_databases/main.py

Este script ejecuta una demostración secuencial de los conceptos clave del módulo:
  1. Embeddings y similitud semántica
  2. ChromaDB — bases de datos vectoriales
  3. RAG básico — pipeline completo
  4. RAG avanzado — multi-query y compresión

Para ejecutar una lección individual:
    python modulo_02_vector_databases/01_embeddings_y_similitud/demo.py
    python modulo_02_vector_databases/02_chromadb_fundamentos/main.py
    python modulo_02_vector_databases/03_rag_basico/main.py
    python modulo_02_vector_databases/04_rag_avanzado/main.py
"""

import subprocess
import sys
from pathlib import Path
from modulo_02_vector_databases.shared.logger import get_logger

MODULE_DIR = Path(__file__).parent

logger = get_logger(__name__)

LECCIONES = [
    {
        "numero": "01",
        "titulo": "Embeddings y Similitud Semántica",
        "script": MODULE_DIR / "01_embeddings_y_similitud" / "demo.py",
    },
    {
        "numero": "02",
        "titulo": "ChromaDB — Bases de Datos Vectoriales",
        "script": MODULE_DIR / "02_chromadb_fundamentos" / "main.py",
    },
    {
        "numero": "03",
        "titulo": "RAG Básico — Retrieval-Augmented Generation",
        "script": MODULE_DIR / "03_rag_basico" / "main.py",
    },
    {
        "numero": "04",
        "titulo": "RAG Avanzado — Multi-Query y Compresión",
        "script": MODULE_DIR / "04_rag_avanzado" / "main.py",
    },
]


def ejecutar_leccion(leccion: dict) -> bool:
    """
    Ejecuta un script de lección como subproceso.

    Returns:
        True si el script terminó sin error, False si falló.
    """
    numero  = leccion["numero"]
    titulo  = leccion["titulo"]
    script  = leccion["script"]

    if not script.exists():
        logger.error(f"Script no encontrado: {script}")
        return False

    separador = "═" * 70
    print(f"\n{separador}")
    print(f"  LECCIÓN {numero}: {titulo}")
    print(f"{separador}\n")

    logger.info(f"Ejecutando lección {numero}: {titulo}")

    try:
        subprocess.run(
            [sys.executable, str(script)],
            check=True,
            text=True,
        )
        logger.info(f"Lección {numero} completada exitosamente")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(
            f"Lección {numero} terminó con error (código {e.returncode}). "
            f"Consulta el output de arriba para detalles."
        )
        return False
    except Exception as e:
        logger.error(f"Error inesperado al ejecutar lección {numero}: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "═" * 70)
    print("  MÓDULO 02: Bases de Datos Vectoriales, Embeddings y RAG")
    print("  Ejecutando todas las lecciones en secuencia...")
    print("═" * 70)

    resultados = []
    for leccion in LECCIONES:
        exito = ejecutar_leccion(leccion)
        resultados.append((leccion["numero"], leccion["titulo"], exito))

    print("\n" + "═" * 70)
    print("  RESUMEN DE EJECUCIÓN")
    print("═" * 70)
    for numero, titulo, exito in resultados:
        estado = "✅ OK" if exito else "❌ FALLÓ"
        print(f"  {estado} — Lección {numero}: {titulo}")

    total   = len(resultados)
    exitosos = sum(1 for _, _, e in resultados if e)
    print(f"\n  {exitosos}/{total} lecciones completadas correctamente")

    if exitosos < total:
        logger.warning(
            f"{total - exitosos} lección(es) fallaron. "
            "Ejecuta cada script individualmente para ver el error completo."
        )
        sys.exit(1)
    else:
        logger.info("Módulo 02 completado. ¡A por el Módulo 03 — LangGraph!")

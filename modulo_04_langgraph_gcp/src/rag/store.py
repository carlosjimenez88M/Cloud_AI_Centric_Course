"""
RAG Store — carga el ChromaDB ya construido en modulo_03_gcp.

Este módulo NO reconstruye el índice: asume que el ChromaDB ya existe
gracias al módulo anterior. Solo lo monta en memoria para hacer búsquedas.

Si el ChromaDB no existe, lanza FileNotFoundError con instrucciones claras.
"""

from __future__ import annotations

import warnings as _warnings
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_google_vertexai import VertexAIEmbeddings

from shared.logger import get_logger
from shared.config import get_module_root

log = get_logger("rag.store")


def load_vector_store(
    cfg: dict[str, Any],
    project_id: str,
    location: str,
) -> Chroma:
    """
    Carga el ChromaDB persistido en modulo_03_gcp y lo devuelve listo para
    hacer búsquedas semánticas.

    Args:
        cfg:        Configuración completa cargada desde config.yaml.
        project_id: ID del proyecto GCP (necesario para VertexAI Embeddings).
        location:   Región de Vertex AI (ej: "us-central1").

    Returns:
        Instancia de Chroma conectada a la colección existente.

    Raises:
        FileNotFoundError: Si el ChromaDB no existe en la ruta esperada.
    """
    # Resuelve la ruta relativa desde la raíz del módulo 04
    root = get_module_root()
    chroma_path = (root / cfg["rag"]["chroma_path"]).resolve()
    collection_name = cfg["rag"]["collection"]

    # Verificación de existencia — mensaje de error amigable para estudiantes
    if not chroma_path.exists():
        raise FileNotFoundError(
            f"\n{'='*60}\n"
            f"ChromaDB no encontrado en:\n  {chroma_path}\n\n"
            f"El ChromaDB se construye en el módulo anterior.\n"
            f"Para generarlo, ejecuta:\n"
            f"  cd modulo_03_gcp\n"
            f"  uv run 02_rag_pipeline.py\n"
            f"{'='*60}\n"
        )

    log.info(
        f"Cargando ChromaDB desde: [dim]{chroma_path}[/dim]"
    )

    # Inicializa el modelo de embeddings de Vertex AI
    # Debe ser el mismo modelo que se usó para construir el índice en modulo_03
    embedding_model = cfg["embedding"]["model"]
    # Suprimir warning de clase deprecated — VertexAIEmbeddings usa Vertex AI (ADC).
    # Usar GoogleGenerativeAIEmbeddings requiere generativelanguage.googleapis.com
    # habilitado en el proyecto, lo cual no está disponible aquí.
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        embeddings = VertexAIEmbeddings(
            model_name=embedding_model,
            project=project_id,
            location=location,
        )
    log.info(f"  Modelo de embeddings: [cyan]{embedding_model}[/cyan]")

    # Conecta al ChromaDB existente (persist_directory = ruta local)
    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(chroma_path),
    )

    # Verifica que el store tiene documentos indexados
    count = store._collection.count()
    if count == 0:
        log.warning(
            "El ChromaDB existe pero está vacío. "
            "Reconstrúyelo ejecutando el módulo 03."
        )
    else:
        log.info(
            f"  [success]ChromaDB cargado[/success] — "
            f"[bold]{count}[/bold] vectores en colección '[cyan]{collection_name}[/cyan]'"
        )

    return store

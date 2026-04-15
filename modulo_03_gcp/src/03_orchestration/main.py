"""
Lección 3: Orquestación Profunda — LangGraph + Routing + Chaining
════════════════════════════════════════════════════════════════════

Requiere que el RAG pipeline (Lección 2) ya haya sido ejecutado al menos
hasta el STEP 3 (ChromaDB indexado).

Ejecución:
    cd modulo_03_gcp
    uv run src/orchestration/main.py
"""

from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_google_vertexai import VertexAIEmbeddings

from shared.logger import get_logger, console
from shared.config_loader import (
    load_config,
    get_project_id,
    get_location,
    get_module_root,
)

from graph import OrchestrationGraph

import warnings as _warnings

log = get_logger("orchestration.main")


def _load_vector_store(
    project_id: str,
    location: str,
    embedding_model: str,
    chroma_path: Path,
    collection_name: str,
) -> Chroma:
    if not chroma_path.exists():
        raise FileNotFoundError(
            f"ChromaDB no encontrado en {chroma_path}.\n"
            "  Ejecuta primero: uv run src/rag/main.py"
        )
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        embeddings = VertexAIEmbeddings(
            model_name=embedding_model,
            project=project_id,
            location=location,
        )
    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(chroma_path),
    )
    count: int = store._collection.count()  # type: ignore[attr-defined]
    log.info(f"ChromaDB cargado — [bold]{count}[/bold] vectores en '{collection_name}'")
    return store


def main() -> None:
    cfg = load_config()
    project_id = get_project_id()
    location = get_location()
    root = get_module_root()

    rag_cfg = cfg["rag"]
    model_name = cfg["model"]["name"]
    emb_model = cfg["embedding"]["model"]
    chroma_path = root / rag_cfg["chroma_path"]

    console.rule("[bold magenta]Orquestación Profunda — Módulo 03[/bold magenta]")
    console.print(f"  Proyecto  : [cyan]{project_id}[/cyan]")
    console.print(f"  Modelo    : [cyan]{model_name}[/cyan]")
    console.print(f"  Pipeline  : Pregunta → [cyan]Router → RAG → Chain → Síntesis[/cyan]")
    console.print()

    vector_store = _load_vector_store(
        project_id=project_id,
        location=location,
        embedding_model=emb_model,
        chroma_path=chroma_path,
        collection_name=rag_cfg["collection"],
    )
    console.print()

    graph = OrchestrationGraph(
        vector_store=vector_store,
        project_id=project_id,
        location=location,
        model_name=model_name,
        top_k=int(rag_cfg["top_k"]),
    )
    console.print()
    graph.run()


if __name__ == "__main__":
    main()

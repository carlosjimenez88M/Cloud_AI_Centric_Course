"""
Lección 3: Orquestación Profunda — LangGraph + Routing + Chaining
════════════════════════════════════════════════════════════════════

Requiere que el RAG pipeline (Lección 2) ya haya sido ejecutado al menos
hasta el STEP 3 (ChromaDB indexado).

Ejecución:
    cd modulo_03_openai
    uv run 03_orchestration.py
"""

from __future__ import annotations

import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from shared.logger import get_logger, console
from shared.config_loader import (
    load_config,
    load_env,
    get_module_root,
)

from graph import OrchestrationGraph

log = get_logger("orchestration.main")


def _load_vector_store(
    embedding_model: str,
    chroma_path: Path,
    collection_name: str,
) -> Chroma:
    if not chroma_path.exists():
        raise FileNotFoundError(
            f"ChromaDB no encontrado en {chroma_path}.\n"
            "  Ejecuta primero: uv run 02_rag_pipeline.py"
        )
    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        api_key=os.getenv("OPENAI_API_KEY"),
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
    load_env()
    cfg = load_config()
    root = get_module_root()

    rag_cfg = cfg["rag"]
    model_name = cfg["model"]["name"]
    emb_model = cfg["embedding"]["model"]
    chroma_path = root / rag_cfg["chroma_path"]

    console.rule("[bold magenta]Orquestación Profunda — Módulo 03[/bold magenta]")
    console.print(f"  Modelo    : [cyan]{model_name}[/cyan]")
    console.print(f"  Pipeline  : Pregunta → [cyan]Router → RAG → Chain → Síntesis[/cyan]")
    console.print()

    vector_store = _load_vector_store(
        embedding_model=emb_model,
        chroma_path=chroma_path,
        collection_name=rag_cfg["collection"],
    )
    console.print()

    graph = OrchestrationGraph(
        vector_store=vector_store,
        model_name=model_name,
        top_k=int(rag_cfg["top_k"]),
    )
    console.print()
    graph.run()


if __name__ == "__main__":
    main()

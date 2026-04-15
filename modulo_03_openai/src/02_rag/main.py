"""
Lección 2: RAG Pipeline — Las Mil y Una Noches (4 pasos)
════════════════════════════════════════════════════════════════════

  STEP 1 — Ingest  : Verifica los PDFs locales listos para procesar
  STEP 2 — Split   : Divide el PDF en chunks semánticos
  STEP 3 — Embed   : Embeddings con OpenAI → ChromaDB local
  STEP 4 — Agent   : Agente LangGraph ReAct con herramienta de recuperación

Ejecución:
    cd modulo_03_openai
    uv run 02_rag_pipeline.py
"""

from __future__ import annotations

from shared.logger import get_logger, console
from shared.config_loader import (
    load_config,
    load_env,
    get_module_root,
)

from step1_ingest import LocalIngestor
from step2_split import DocumentSplitter
from step3_embed import EmbeddingIndexer
from step4_agent import AgenticRAG

log = get_logger("rag.main")


def main() -> None:
    load_env()
    cfg = load_config()
    root = get_module_root()

    rag_cfg = cfg["rag"]
    model_name = cfg["model"]["name"]
    emb_model = cfg["embedding"]["model"]

    database_path = root / rag_cfg["database_path"]
    chroma_path = root / rag_cfg["chroma_path"]

    console.rule("[bold magenta]RAG Pipeline — Las Mil y Una Noches[/bold magenta]")
    console.print(f"  Modelo LLM: [cyan]{model_name}[/cyan]")
    console.print(f"  Embeddings: [cyan]{emb_model}[/cyan]")
    console.print(f"  Documentos: [cyan]{database_path}[/cyan]")
    console.print()

    # ── STEP 1: Ingest (local) ─────────────────────────────────────────────────
    ingestor = LocalIngestor()
    ingestor.run(database_path)
    console.print()

    # ── STEP 2: Split ──────────────────────────────────────────────────────────
    splitter = DocumentSplitter(
        chunk_size=int(rag_cfg["chunk_size"]),
        chunk_overlap=int(rag_cfg["chunk_overlap"]),
    )
    chunks = splitter.run(database_path, max_pages=int(rag_cfg.get("max_pages", 0)))
    console.print()

    # ── STEP 3: Embed ─────────────────────────────────────────────────────────
    indexer = EmbeddingIndexer(
        embedding_model=emb_model,
        chroma_path=chroma_path,
        collection_name=rag_cfg["collection"],
    )
    vector_store = indexer.run(chunks)
    console.print()

    # ── STEP 4: Agentic RAG ────────────────────────────────────────────────────
    rag = AgenticRAG(
        vector_store=vector_store,
        model_name=model_name,
        top_k=int(rag_cfg["top_k"]),
    )
    rag.run()

    # ── Resumen ────────────────────────────────────────────────────────────────
    console.print()
    console.rule("[bold green]Pipeline completado[/bold green]")
    console.print(f"  ChromaDB local : [cyan]{chroma_path}[/cyan]")


if __name__ == "__main__":
    main()

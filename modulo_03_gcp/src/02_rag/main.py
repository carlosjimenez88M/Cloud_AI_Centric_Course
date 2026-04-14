"""
Lección 2: RAG Pipeline — Las Mil y Una Noches (4 pasos)
════════════════════════════════════════════════════════════════════

  STEP 1 — Ingest  : Sube el PDF a Google Cloud Storage
  STEP 2 — Split   : Divide el PDF en chunks semánticos
  STEP 3 — Embed   : Embeddings con Vertex AI → ChromaDB local + sync a GCS
  STEP 4 — Agent   : Agente LangGraph ReAct con herramienta de recuperación

Ejecución:
    cd modulo_03_gcp
    uv run src/rag/main.py
"""

from __future__ import annotations

from shared.logger import get_logger, console
from shared.config_loader import (
    load_config,
    get_project_id,
    get_location,
    get_module_root,
)

from step1_ingest import GCSIngestor
from step2_split import DocumentSplitter
from step3_embed import EmbeddingIndexer
from step4_agent import AgenticRAG

log = get_logger("rag.main")


def main() -> None:
    cfg = load_config()
    project_id = get_project_id()
    location = get_location()
    root = get_module_root()

    rag_cfg = cfg["rag"]
    model_name = cfg["model"]["name"]
    emb_model = cfg["embedding"]["model"]

    database_path = root / rag_cfg["database_path"]
    chroma_path = root / rag_cfg["chroma_path"]

    console.rule("[bold magenta]RAG Pipeline — Las Mil y Una Noches[/bold magenta]")
    console.print(f"  Proyecto  : [cyan]{project_id}[/cyan]")
    console.print(f"  Modelo LLM: [cyan]{model_name}[/cyan]")
    console.print(f"  Embeddings: [cyan]{emb_model}[/cyan]")
    console.print(
        f"  Bucket    : [cyan]gs://{rag_cfg['bucket']}/{rag_cfg['bucket_prefix']}[/cyan]"
    )
    console.print()

    # ── STEP 1: Ingest ─────────────────────────────────────────────────────────
    ingestor = GCSIngestor(
        bucket_name=rag_cfg["bucket"],
        prefix=rag_cfg["bucket_prefix"],
        project_id=project_id,
    )
    gcs_uris = ingestor.run(database_path)
    console.print()

    # ── STEP 2: Split ──────────────────────────────────────────────────────────
    splitter = DocumentSplitter(
        chunk_size=int(rag_cfg["chunk_size"]),
        chunk_overlap=int(rag_cfg["chunk_overlap"]),
    )
    chunks = splitter.run(database_path, max_pages=int(rag_cfg.get("max_pages", 0)))
    console.print()

    # ── STEP 3: Embed + sync a GCS ────────────────────────────────────────────
    indexer = EmbeddingIndexer(
        project_id=project_id,
        location=location,
        embedding_model=emb_model,
        chroma_path=chroma_path,
        collection_name=rag_cfg["collection"],
    )
    vector_store = indexer.run(chunks)

    # Sync ChromaDB a GCS si está habilitado en config.yaml
    if rag_cfg.get("sync_chroma_gcs", False):
        chroma_gcs_uri = indexer.sync_to_gcs(
            bucket_name=rag_cfg["bucket"],
            gcs_prefix=rag_cfg.get("chroma_gcs_prefix", "modulo_03_rag/chroma_db/"),
            project_id=project_id,
        )
        console.print(
            f"  [bold green]ChromaDB visible en GCS:[/bold green] [cyan]{chroma_gcs_uri}[/cyan]"
        )
    console.print()

    # ── STEP 4: Agentic RAG ────────────────────────────────────────────────────
    rag = AgenticRAG(
        vector_store=vector_store,
        project_id=project_id,
        location=location,
        model_name=model_name,
        top_k=int(rag_cfg["top_k"]),
    )
    rag.run()

    # ── Resumen ────────────────────────────────────────────────────────────────
    console.print()
    console.rule("[bold green]Pipeline completado[/bold green]")
    console.print(f"  ChromaDB local : [cyan]{chroma_path}[/cyan]")
    for uri in gcs_uris:
        console.print(f"  Documento GCS  : [cyan]{uri}[/cyan]")


if __name__ == "__main__":
    main()

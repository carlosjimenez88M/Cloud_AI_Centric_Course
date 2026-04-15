"""
STEP 3 — Embedding Indexer
━━━━━━━━━━━━━━━━━━━━━━━━━━
Responsabilidad única: generar embeddings con OpenAI (text-embedding-3-small)
y persistirlos en ChromaDB para recuperación eficiente.

Idempotente — si la colección ya tiene tantos o más vectores que los chunks
provistos, no re-indexa.

Uso independiente:
    from rag.step3_embed import EmbeddingIndexer
    indexer = EmbeddingIndexer(
        embedding_model="text-embedding-3-small",
        chroma_path=Path("data/chroma_db"),
        collection_name="mily_una_noches",
    )
    vector_store = indexer.run(chunks)
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from shared.logger import get_logger

log = get_logger("rag.step3")

# Tamaño de lote para llamadas a la API de embeddings.
# text-embedding-3-small admite hasta 2048 inputs por request.
# Con chunks de ~1000 chars, 50 chunks por lote es seguro y eficiente.
_BATCH_SIZE = 50


class EmbeddingIndexer:
    """
    Genera embeddings con OpenAI y los guarda en ChromaDB.

    Args:
        embedding_model: Nombre del modelo de embeddings OpenAI.
        chroma_path:     Ruta para persistir ChromaDB.
        collection_name: Nombre de la colección en ChromaDB.
    """

    def __init__(
        self,
        embedding_model: str,
        chroma_path: Path,
        collection_name: str,
    ) -> None:
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        chroma_path.mkdir(parents=True, exist_ok=True)

        self._embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        log.info(
            f"EmbeddingIndexer — modelo: [cyan]{embedding_model}[/cyan]  "
            f"colección: [cyan]{collection_name}[/cyan]"
        )

    def _open_store(self) -> Chroma:
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self._embeddings,
            persist_directory=str(self.chroma_path),
        )

    def run(self, chunks: list[Document]) -> Chroma:
        """
        Indexa los chunks en ChromaDB.

        Si la colección ya contiene tantos o más vectores, retorna el store
        existente sin re-indexar (comportamiento idempotente).

        Returns:
            Instancia de Chroma lista para consultas.
        """
        log.info(
            f"[step]STEP 3 — Embed:[/step] "
            f"{len(chunks)} chunks → ChromaDB ({self.collection_name})"
        )
        store = self._open_store()
        existing: int = store._collection.count()  # type: ignore[attr-defined]

        if existing >= len(chunks):
            log.info(
                f"  Colección existente con {existing} vectores — "
                "omitiendo re-indexación"
            )
            log.info("[success]STEP 3 OK (cache hit)[/success]")
            return store

        n_batches = -(-len(chunks) // _BATCH_SIZE)  # ceil division
        log.info(
            f"  Indexando {len(chunks)} chunks en {n_batches} lotes "
            f"de {_BATCH_SIZE}…"
        )
        t0 = time.perf_counter()
        for batch_num, start in enumerate(range(0, len(chunks), _BATCH_SIZE), 1):
            batch = chunks[start : start + _BATCH_SIZE]
            store.add_documents(batch)
            log.info(f"  Lote {batch_num}/{n_batches} ({len(batch)} chunks)")

        elapsed = round(time.perf_counter() - t0, 1)
        total: int = store._collection.count()  # type: ignore[attr-defined]
        log.info(
            f"[success]STEP 3 OK — {total} vectores persistidos "
            f"en {elapsed}s[/success]"
        )
        return store

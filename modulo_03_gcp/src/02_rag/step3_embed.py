"""
STEP 3 — Embedding Indexer
━━━━━━━━━━━━━━━━━━━━━━━━━━
Responsabilidad única: generar embeddings con Vertex AI (text-embedding-004)
y persistirlos en ChromaDB para recuperación eficiente.

Idempotente — si la colección ya tiene tantos o más vectores que los chunks
provistos, no re-indexa.

Uso independiente:
    from rag.step3_embed import EmbeddingIndexer
    indexer = EmbeddingIndexer(
        project_id="...", location="us-central1",
        embedding_model="text-embedding-004",
        chroma_path=Path("data/chroma_db"),
        collection_name="mily_una_noches",
    )
    vector_store = indexer.run(chunks)
"""

from __future__ import annotations

import time
from pathlib import Path

from google.cloud import storage as gcs
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_vertexai import VertexAIEmbeddings
from shared.logger import get_logger

log = get_logger("rag.step3")

# Tamaño de lote para llamadas a la API de embeddings.
# text-embedding-004 admite máx. 20 000 tokens por lote.
# Con chunks de ~1 000 chars (~400 tokens), 20 chunks ≈ 8 000 tokens → margen seguro.
_BATCH_SIZE = 20


class EmbeddingIndexer:
    """
    Genera embeddings con Vertex AI y los guarda en ChromaDB.

    Args:
        project_id:      ID del proyecto GCP.
        location:        Región de Vertex AI (ej. "us-central1").
        embedding_model: Nombre del modelo de embeddings.
        chroma_path:     Ruta para persistir ChromaDB.
        collection_name: Nombre de la colección en ChromaDB.
    """

    def __init__(
        self,
        project_id: str,
        location: str,
        embedding_model: str,
        chroma_path: Path,
        collection_name: str,
    ) -> None:
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        chroma_path.mkdir(parents=True, exist_ok=True)

        self._embeddings = VertexAIEmbeddings(
            model_name=embedding_model,
            project=project_id,
            location=location,
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

    # ── GCS Sync ──────────────────────────────────────────────────────────────

    def sync_to_gcs(
        self,
        bucket_name: str,
        gcs_prefix: str,
        project_id: str,
    ) -> str:
        """
        Sincroniza el directorio ChromaDB local a Google Cloud Storage.

        Los archivos son **navegables** en GCS Console:
          Cloud Storage → Buckets → <bucket> → <gcs_prefix>

        Args:
            bucket_name: Nombre del bucket GCS.
            gcs_prefix:  Prefijo de destino (ej. "modulo_03_rag/chroma_db/").
            project_id:  ID del proyecto GCP.

        Returns:
            URI base: gs://<bucket>/<gcs_prefix>
        """
        prefix = gcs_prefix.rstrip("/") + "/"
        client = gcs.Client(project=project_id)
        bucket = client.bucket(bucket_name)

        files = [f for f in self.chroma_path.rglob("*") if f.is_file()]
        log.info(
            f"[step]GCS Sync:[/step] subiendo {len(files)} archivo(s) de ChromaDB → "
            f"gs://{bucket_name}/{prefix}"
        )
        for local_file in files:
            relative = local_file.relative_to(self.chroma_path)
            blob_name = prefix + str(relative).replace("\\", "/")
            bucket.blob(blob_name).upload_from_filename(str(local_file))
            log.info(f"  ↑ {relative}")

        base_uri = f"gs://{bucket_name}/{prefix}"
        log.info(f"[success]GCS Sync OK — {base_uri}[/success]")
        return base_uri

    def sync_from_gcs(
        self,
        bucket_name: str,
        gcs_prefix: str,
        project_id: str,
    ) -> bool:
        """
        Descarga ChromaDB desde GCS si existe. Útil para restaurar el índice
        en un nuevo entorno sin re-indexar.

        Returns:
            True si se descargó algo, False si no había nada en GCS.
        """
        prefix = gcs_prefix.rstrip("/") + "/"
        client = gcs.Client(project=project_id)
        blobs = list(client.list_blobs(bucket_name, prefix=prefix))
        if not blobs:
            log.info(f"  No hay ChromaDB en GCS bajo {prefix} — se indexará localmente")
            return False

        log.info(
            f"[step]GCS Restore:[/step] descargando {len(blobs)} archivo(s) desde "
            f"gs://{bucket_name}/{prefix}"
        )
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        for blob in blobs:
            relative = blob.name[len(prefix):]
            if not relative:
                continue
            local_file = self.chroma_path / relative
            local_file.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(local_file))
            log.info(f"  ↓ {relative}")

        log.info(f"[success]GCS Restore OK — ChromaDB restaurado en {self.chroma_path}[/success]")
        return True

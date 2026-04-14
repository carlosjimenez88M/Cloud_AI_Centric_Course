"""
STEP 1 — GCS Ingestor
━━━━━━━━━━━━━━━━━━━━━
Responsabilidad única: subir documentos locales a un bucket de Google Cloud Storage.
Idempotente — no re-sube archivos que ya existen en el bucket.

Uso independiente:
    from rag.step1_ingest import GCSIngestor
    ingestor = GCSIngestor(bucket_name="...", prefix="modulo_03_rag/", project_id="...")
    uris = ingestor.run(Path("src/database"))
"""

from __future__ import annotations

from pathlib import Path

from google.cloud import storage as gcs
from shared.logger import get_logger

log = get_logger("rag.step1")


class GCSIngestor:
    """
    Sube documentos desde un directorio local a Google Cloud Storage.

    Args:
        bucket_name: Nombre del bucket GCS (debe existir).
        prefix:      Prefijo/carpeta dentro del bucket (ej. "modulo_03_rag/").
        project_id:  ID del proyecto GCP.
    """

    def __init__(self, bucket_name: str, prefix: str, project_id: str) -> None:
        self.bucket_name = bucket_name
        self.prefix = prefix.rstrip("/") + "/"
        self._client = gcs.Client(project=project_id)
        self._bucket = self._client.bucket(bucket_name)
        log.info(f"GCSIngestor → [cyan]gs://{bucket_name}/{self.prefix}[/cyan]")

    def upload_file(self, local_path: Path) -> str:
        """
        Sube un archivo al bucket.

        Returns:
            URI completa: gs://<bucket>/<prefix>/<filename>
        """
        blob_name = self.prefix + local_path.name
        blob = self._bucket.blob(blob_name)
        uri = f"gs://{self.bucket_name}/{blob_name}"

        if blob.exists():
            size_kb = blob.size // 1024 if blob.size else "?"
            log.info(f"  [dim]Existe ({size_kb} KB) — omitiendo: {local_path.name}[/dim]")
        else:
            size_kb = local_path.stat().st_size // 1024
            log.info(f"  Subiendo {local_path.name} ({size_kb} KB)…")
            blob.upload_from_filename(str(local_path))
            log.info(f"  [green]Subido:[/green] {uri}")

        return uri

    def run(self, database_path: Path, extensions: tuple[str, ...] = (".pdf",)) -> list[str]:
        """
        Sube todos los archivos con las extensiones indicadas.

        Args:
            database_path: Directorio local con los documentos.
            extensions:    Tupla de extensiones a subir (por defecto .pdf).

        Returns:
            Lista de URIs gs:// subidas.
        """
        files = sorted(
            f for ext in extensions for f in database_path.glob(f"*{ext}")
        )
        if not files:
            raise FileNotFoundError(
                f"No se encontraron archivos {extensions} en {database_path}"
            )

        log.info(
            f"[step]STEP 1 — Ingest:[/step] "
            f"{len(files)} archivo(s) → gs://{self.bucket_name}/{self.prefix}"
        )
        uris = [self.upload_file(f) for f in files]
        log.info(f"[success]STEP 1 OK — {len(uris)} URI(s) en GCS[/success]")
        return uris

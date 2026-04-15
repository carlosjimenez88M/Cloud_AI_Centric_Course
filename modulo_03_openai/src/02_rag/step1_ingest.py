"""
STEP 1 — Local Ingestor
━━━━━━━━━━━━━━━━━━━━━━━
Responsabilidad única: verificar y listar documentos locales listos
para el pipeline. Sin dependencia de cloud storage.

Uso independiente:
    from rag.step1_ingest import LocalIngestor
    ingestor = LocalIngestor()
    paths = ingestor.run(Path("src/database"))
"""

from __future__ import annotations

from pathlib import Path

from shared.logger import get_logger

log = get_logger("rag.step1")


class LocalIngestor:
    """
    Verifica y lista documentos locales para el pipeline RAG.
    Reemplaza GCSIngestor — no requiere cloud storage.
    """

    def __init__(self) -> None:
        log.info("LocalIngestor — procesamiento local (sin GCS)")

    def run(self, database_path: Path, extensions: tuple[str, ...] = (".pdf",)) -> list[Path]:
        """
        Lista todos los archivos con las extensiones indicadas.

        Args:
            database_path: Directorio local con los documentos.
            extensions:    Tupla de extensiones a procesar (por defecto .pdf).

        Returns:
            Lista de Path de los archivos encontrados.
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
            f"{len(files)} archivo(s) encontrado(s) en {database_path}"
        )
        for f in files:
            size_kb = f.stat().st_size // 1024
            log.info(f"  [green]Listo:[/green] {f.name} ({size_kb} KB)")

        log.info(f"[success]STEP 1 OK — {len(files)} archivo(s) listos para procesar[/success]")
        return files

"""
STEP 2 — Document Splitter
━━━━━━━━━━━━━━━━━━━━━━━━━━
Responsabilidad única: cargar PDFs localmente y dividirlos en chunks semánticos
usando RecursiveCharacterTextSplitter de LangChain.

Uso independiente:
    from rag.step2_split import DocumentSplitter
    splitter = DocumentSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.run(Path("src/database"))
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from shared.logger import get_logger

log = get_logger("rag.step2")


class DocumentSplitter:
    """
    Carga documentos PDF y los divide en chunks con metadata enriquecida.

    Args:
        chunk_size:    Tamaño máximo de cada chunk en caracteres.
        chunk_overlap: Solapamiento entre chunks consecutivos.
    """

    _SEPARATORS = ["\n\n", "\n", ".", "?", "!", ";", " "]

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self._SEPARATORS,
            length_function=len,
        )
        log.info(
            f"DocumentSplitter — "
            f"chunk_size={chunk_size}  overlap={chunk_overlap}"
        )

    def split_pdf(self, pdf_path: Path, max_pages: int = 0) -> list[Document]:
        """
        Carga y divide un PDF en chunks con metadata extendida.

        Args:
            max_pages: Límite de páginas a procesar (0 = sin límite).

        Returns:
            Lista de Document (LangChain) con metadata:
              source_file, chunk_index, total_chunks, page.
        """
        log.info(f"  Cargando {pdf_path.name} …")
        pages = PyPDFLoader(str(pdf_path)).load()
        if max_pages and max_pages < len(pages):
            log.info(
                f"  Páginas totales: {len(pages)} — "
                f"limitando a {max_pages} (modo demo)"
            )
            pages = pages[:max_pages]
        else:
            log.info(f"  Páginas: {len(pages)}")

        chunks = self._splitter.split_documents(pages)
        for i, chunk in enumerate(chunks):
            chunk.metadata.update(
                {
                    "source_file": pdf_path.name,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }
            )

        log.info(f"  Chunks generados: [bold]{len(chunks)}[/bold]")
        return chunks

    def run(self, database_path: Path, max_pages: int = 0) -> list[Document]:
        """
        Divide todos los PDFs del directorio.

        Args:
            max_pages: Límite de páginas por documento (0 = sin límite).

        Returns:
            Lista consolidada de chunks de todos los documentos.
        """
        pdfs = sorted(database_path.glob("*.pdf"))
        if not pdfs:
            raise FileNotFoundError(f"No hay PDFs en {database_path}")

        log.info(
            f"[step]STEP 2 — Split:[/step] "
            f"{len(pdfs)} PDF(s) en {database_path.name}/"
        )
        all_chunks: list[Document] = []
        for pdf in pdfs:
            all_chunks.extend(self.split_pdf(pdf, max_pages=max_pages))

        log.info(
            f"[success]STEP 2 OK — {len(all_chunks)} chunks totales "
            f"de {len(pdfs)} documento(s)[/success]"
        )
        return all_chunks

"""
rag — Módulo RAG para Google Cloud (modulo_03_gcp)

Pasos del pipeline:
  from rag.step1_ingest import GCSIngestor
  from rag.step2_split  import DocumentSplitter
  from rag.step3_embed  import EmbeddingIndexer
  from rag.step4_agent  import AgenticRAG
"""
from .step1_ingest import GCSIngestor
from .step2_split import DocumentSplitter
from .step3_embed import EmbeddingIndexer
from .step4_agent import AgenticRAG

__all__ = ["GCSIngestor", "DocumentSplitter", "EmbeddingIndexer", "AgenticRAG"]

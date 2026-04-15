"""
rag — Módulo RAG para OpenAI (modulo_03_openai)

Pasos del pipeline:
  from rag.step1_ingest import LocalIngestor
  from rag.step2_split  import DocumentSplitter
  from rag.step3_embed  import EmbeddingIndexer
  from rag.step4_agent  import AgenticRAG
"""
from step1_ingest import LocalIngestor
from step2_split import DocumentSplitter
from step3_embed import EmbeddingIndexer
from step4_agent import AgenticRAG

__all__ = ["LocalIngestor", "DocumentSplitter", "EmbeddingIndexer", "AgenticRAG"]

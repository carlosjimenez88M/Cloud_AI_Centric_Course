"""RAG module — carga el ChromaDB construido en modulo_03_gcp."""
from .store import load_vector_store

__all__ = ["load_vector_store"]

"""
LECCIÓN 2 — RAG Pipeline (4 pasos)
Las Mil y Una Noches: GCS → Split → Embeddings → Agente LangGraph

Cómo correr:
    cd modulo_03_gcp
    uv run 02_rag_pipeline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

# Agrega src/02_rag al path para imports dentro del módulo
sys.path.insert(0, str(Path(__file__).parent / "src" / "02_rag"))

from main import main   # src/02_rag/main.py

if __name__ == "__main__":
    main()

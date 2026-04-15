"""
LECCIÓN 2 — RAG Pipeline (4 pasos)
Las Mil y Una Noches: Local → Split → Embeddings → Agente LangGraph

Cómo correr:
    cd modulo_03_openai
    uv run 02_rag_pipeline.py
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "src" / "02_rag"))

import main as _m
_m.main()

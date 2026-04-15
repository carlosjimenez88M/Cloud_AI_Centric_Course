"""
LECCIÓN 3 — Orquestación Profunda
Router → RAG → Chain (story/character/philosophy/creative) → Síntesis

Requiere que la Lección 2 se haya ejecutado (ChromaDB ya indexado).

Cómo correr:
    cd modulo_03_gcp
    uv run 03_orchestration.py
"""
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GRPC_TRACE", "")

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "src" / "03_orchestration"))

import main as _m
_m.main()

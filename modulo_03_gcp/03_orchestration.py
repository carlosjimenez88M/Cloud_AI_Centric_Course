"""
LECCIÓN 3 — Orquestación Profunda
Router → RAG → Chain (story/character/philosophy/creative) → Síntesis

Requiere que la Lección 2 se haya ejecutado (ChromaDB ya indexado).

Cómo correr:
    cd modulo_03_gcp
    uv run 03_orchestration.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "src" / "03_orchestration"))

from main import main   # src/03_orchestration/main.py

if __name__ == "__main__":
    main()

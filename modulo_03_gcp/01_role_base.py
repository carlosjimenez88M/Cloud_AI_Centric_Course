"""
LECCIÓN 1 — Role-Based Prompting
Los 4 Fantásticos + Ricardo Arjona con Vertex AI (gemini-2.5-flash-lite)

Cómo correr:
    cd modulo_03_gcp
    uv run 01_role_base.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src" / "01_role_base"))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from main import main  # src/01_role_base/main.py

if __name__ == "__main__":
    main()

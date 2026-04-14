"""
LECCIÓN 1 — Role-Based Prompting
Los 4 Fantásticos + Ricardo Arjona con Vertex AI (gemini-2.5-flash-lite)

Cómo correr:
    cd modulo_03_gcp
    uv run 01_role_base.py
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "src" / "01_role_base"))

import main as _m
_m.main()

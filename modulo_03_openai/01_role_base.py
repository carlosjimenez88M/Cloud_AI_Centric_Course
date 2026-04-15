"""
LECCIÓN 1 — Role-Based Prompting
Los 4 Fantásticos + Ricardo Arjona con OpenAI (gpt-4o-mini)

Cómo correr:
    cd modulo_03_openai
    uv run 01_role_base.py
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "src" / "01_role_base"))

import main as _m
_m.main()

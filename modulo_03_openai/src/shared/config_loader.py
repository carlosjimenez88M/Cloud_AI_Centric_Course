"""
Config loader — reads config.yaml (project root) + .env (src/).
Provides typed helpers used by every script in modulo_03_openai.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# ── Path anchors ──────────────────────────────────────────────────────────────
# This file lives at  modulo_03_openai/src/shared/config_loader.py
# Going up 3 levels   → modulo_03_openai/
_MODULE_ROOT = Path(__file__).resolve().parent.parent.parent
_ENV_FILE = _MODULE_ROOT / "src" / ".env"
_CONFIG_FILE = _MODULE_ROOT / "config.yaml"

_config_cache: dict[str, Any] | None = None


def load_env() -> None:
    """Load .env into process environment (idempotent)."""
    load_dotenv(_ENV_FILE, override=False)


def load_config() -> dict[str, Any]:
    """Load and cache config.yaml."""
    global _config_cache
    if _config_cache is None:
        if not _CONFIG_FILE.exists():
            raise FileNotFoundError(f"config.yaml not found at {_CONFIG_FILE}")
        with open(_CONFIG_FILE) as f:
            _config_cache = yaml.safe_load(f) or {}
    return _config_cache


def get_openai_api_key() -> str:
    load_env()
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set.\n"
            f"  Add  OPENAI_API_KEY=sk-...  to  {_ENV_FILE}"
        )
    return key


def get_model_name() -> str:
    return load_config()["model"]["name"]


def get_module_root() -> Path:
    return _MODULE_ROOT

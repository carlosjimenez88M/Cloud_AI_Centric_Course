"""
Config loader — reads config.yaml (project root) + .env (src/).
Provides typed helpers used by every script in modulo_04_langgraph_gcp.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# ── Path anchors ──────────────────────────────────────────────────────────────
# This file lives at  modulo_04_langgraph_gcp/src/shared/config.py
# Going up 3 levels   → modulo_04_langgraph_gcp/
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


def get_project_id() -> str:
    load_env()
    pid = os.getenv("PROJECT_ID", "")
    if not pid:
        raise EnvironmentError(
            "PROJECT_ID is not set.\n"
            f"  Add  PROJECT_ID=<your-gcp-project>  to  {_ENV_FILE}"
        )
    return pid


def get_location() -> str:
    load_env()
    return os.getenv("LOCATION", load_config().get("model", {}).get("location", "us-central1"))


def get_model_name() -> str:
    return load_config()["model"]["name"]


def get_module_root() -> Path:
    """Return the root path of modulo_04_langgraph_gcp."""
    return _MODULE_ROOT


# ── Esquema de validación del config.yaml ────────────────────────────────────
_REQUIRED: dict[str, list[str]] = {
    "model":        ["name", "temperature", "max_output_tokens"],
    "embedding":    ["model"],
    "rag":          ["chroma_path", "collection", "top_k"],
    "multi_agent":  ["max_iterations"],
}


def validate_config(cfg: dict[str, Any]) -> list[str]:
    """
    Valida que config.yaml tiene todas las claves necesarias.

    Returns:
        Lista de errores encontrados (vacía = config válido).
    """
    errors: list[str] = []
    for section, keys in _REQUIRED.items():
        if section not in cfg:
            errors.append(f"Sección '{section}' faltante en config.yaml")
            continue
        for key in keys:
            if key not in cfg[section]:
                errors.append(f"Clave '{section}.{key}' faltante en config.yaml")
    return errors

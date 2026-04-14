"""Shared utilities for modulo_03_gcp."""
from .logger import get_logger, console
from .config_loader import load_config, load_env, get_project_id

__all__ = ["get_logger", "console", "load_config", "load_env", "get_project_id"]

"""
Colored logger using Rich — shared across all scripts in modulo_03_openai.

Usage:
    from shared.logger import get_logger, console
    log = get_logger(__name__)
    log.info("Starting pipeline")
    console.print("[bold green]Done![/bold green]")
"""

import logging
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme

# ── Custom color theme ────────────────────────────────────────────────────────
_THEME = Theme(
    {
        "logging.level.info": "bold cyan",
        "logging.level.warning": "bold yellow",
        "logging.level.error": "bold red",
        "logging.level.critical": "bold white on red",
        "logging.level.debug": "dim white",
        "step": "bold blue",
        "success": "bold green",
        "highlight": "bold magenta",
        "dim": "dim white",
    }
)

console = Console(theme=_THEME)

# ── Registry to avoid duplicate handlers ────────────────────────────────────
_loggers: dict[str, logging.Logger] = {}


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a Rich-colored logger.  Safe to call multiple times with the same
    name — the same instance is returned without adding duplicate handlers.
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            markup=True,
            show_path=False,
            show_time=True,
        )
        handler.setLevel(level)
        logger.addHandler(handler)

    _loggers[name] = logger
    return logger

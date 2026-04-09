"""
shared/logger.py — Logger con colores ANSI para el Módulo 02

Por qué usar un logger con colores:
  En producción y en aprendizaje, necesitamos distinguir rápidamente entre
  mensajes informativos, advertencias y errores. El color reduce la carga
  cognitiva: no tienes que leer cada línea, el color te indica si algo
  salió mal de un vistazo.

Uso:
    from shared.logger import get_logger
    logger = get_logger(__name__)

    logger.debug("Detalle interno, normalmente oculto")
    logger.info("Operación normal completada")
    logger.warning("Algo merece atención pero no es un error")
    logger.error("Algo falló, pero el programa puede continuar")
    logger.critical("Fallo crítico, el sistema está comprometido")
"""

import logging
import sys


class ColoredFormatter(logging.Formatter):
    """
    Formateador de logs con colores ANSI.

    Paleta de colores por nivel:
      DEBUG    → Cyan    (información de depuración, poco visible a propósito)
      INFO     → Verde   (operaciones normales y correctas)
      WARNING  → Amarillo (situaciones que merecen revisión)
      ERROR    → Rojo    (fallos recuperables)
      CRITICAL → Magenta (fallos que comprometen el sistema)

    Nota: Los códigos ANSI solo funcionan en terminales que los soporten
    (macOS Terminal, iTerm2, VS Code terminal, Linux shells). En Windows
    puede requerir el módulo `colorama` o usar Windows Terminal.
    """

    # Códigos de escape ANSI para colores y estilos
    _RESET  = "\033[0m"
    _BOLD   = "\033[1m"
    _DIM    = "\033[2m"

    _COLORS = {
        logging.DEBUG:    "\033[36m",    # Cyan
        logging.INFO:     "\033[32m",    # Verde
        logging.WARNING:  "\033[33m",    # Amarillo
        logging.ERROR:    "\033[31m",    # Rojo
        logging.CRITICAL: "\033[35m",    # Magenta
    }

    # Nombres de nivel alineados para que la salida quede ordenada
    _LEVEL_NAMES = {
        logging.DEBUG:    "DEBUG   ",
        logging.INFO:     "INFO    ",
        logging.WARNING:  "WARNING ",
        logging.ERROR:    "ERROR   ",
        logging.CRITICAL: "CRITICAL",
    }

    def format(self, record: logging.LogRecord) -> str:
        color    = self._COLORS.get(record.levelno, self._RESET)
        level    = self._LEVEL_NAMES.get(record.levelno, record.levelname)
        ts       = self.formatTime(record, "%H:%M:%S")
        message  = record.getMessage()

        # Truncar nombre de módulo a 28 chars para mantener el layout limpio
        module = record.name[:28].ljust(28)

        line = (
            f"{self._DIM}{ts}{self._RESET}"
            f" │ {self._DIM}{module}{self._RESET}"
            f" │ {self._BOLD}{color}{level}{self._RESET}"
            f" │ {color}{message}{self._RESET}"
        )

        # Agregar traceback si hay excepción
        if record.exc_info:
            exc_text = self.formatException(record.exc_info)
            line = f"{line}\n{self._COLORS[logging.ERROR]}{exc_text}{self._RESET}"

        return line


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Crea y retorna un logger con salida coloreada.

    Args:
        name:  Nombre del logger. Convención: pasar siempre __name__
               para que el log indique qué módulo lo emitió.
        level: Nivel mínimo a mostrar (default: DEBUG = mostrar todo).
               En producción se usaría logging.INFO o logging.WARNING.

    Returns:
        Logger configurado con handler de consola coloreado.

    Example:
        logger = get_logger(__name__)
        logger.info("Embeddings cargados correctamente")
        logger.error("Fallo al conectar con ChromaDB")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Evitar duplicar handlers si get_logger se llama más de una vez
    # (esto ocurre cuando el módulo se importa en un contexto interactivo)
    if logger.handlers:
        return logger

    # No propagar al root logger para evitar mensajes duplicados
    logger.propagate = False

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(ColoredFormatter())
    logger.addHandler(handler)

    return logger

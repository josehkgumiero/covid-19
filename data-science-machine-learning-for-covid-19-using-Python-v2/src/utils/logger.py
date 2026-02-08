"""
logger.py

Configuração centralizada de logging para o projeto.
Evita duplicação de logs em Jupyter, scripts e execução repetida.
"""

import logging
from pathlib import Path


LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """
    Cria e retorna um logger configurado, sem duplicação de handlers.

    Args:
        name (str): Nome do módulo

    Returns:
        logging.Logger: Logger configurado
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 🔴 CRÍTICO: evita propagação para o root logger
    logger.propagate = False

    # 🔴 Remove handlers existentes (Jupyter / autoreload safe)
    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(LOG_DIR / "application.log")
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

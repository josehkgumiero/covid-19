"""
_setup.py

Configura dinamicamente o PYTHONPATH para permitir a importação
de módulos localizados na pasta /src e em suas camadas internas
(ingestion, cleaning, transformation, utils).

Este script deve ser executado antes de qualquer import nos notebooks.
"""

from pathlib import Path
import sys
import logging
from typing import List


# =========================
# Logging configuration
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


def add_paths_to_pythonpath(paths: List[Path]) -> None:
    """
    Adiciona múltiplos caminhos ao PYTHONPATH, se ainda não existirem.

    Args:
        paths (List[Path]): Lista de caminhos a serem adicionados.
    """
    for path in paths:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
            logger.info("Added to PYTHONPATH: %s", path)
        else:
            logger.debug("Already in PYTHONPATH: %s", path)


def add_src_layers_to_path() -> None:
    """
    Localiza a pasta /src e adiciona suas principais camadas
    ao PYTHONPATH.
    """
    try:
        current_file: Path = Path(__file__).resolve()

        # notebook/_setup.py → notebook → project root
        project_root: Path = current_file.parent.parent
        src_path: Path = project_root / "src"

        if not src_path.exists():
            raise FileNotFoundError(f"src directory not found at: {src_path}")

        # Camadas do projeto (conforme arquitetura)
        layer_dirs: List[Path] = [
            src_path,
            src_path / "ingestion",
            src_path / "cleaning",
            src_path / "transformation",
            src_path / "utils",
        ]

        # Valida existência das camadas
        existing_layers: List[Path] = [
            layer for layer in layer_dirs if layer.exists()
        ]

        if not existing_layers:
            raise RuntimeError("No valid src layers found to add to PYTHONPATH.")

        add_paths_to_pythonpath(existing_layers)

    except Exception as error:
        logger.error(
            "Failed to configure PYTHONPATH for src layers: %s",
            error,
            exc_info=True
        )
        raise RuntimeError(
            "Error while configuring PYTHONPATH for project src layers."
        ) from error


# Executa automaticamente quando importado
add_src_layers_to_path()

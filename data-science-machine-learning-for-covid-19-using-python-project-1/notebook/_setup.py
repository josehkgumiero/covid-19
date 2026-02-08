"""
_setup.py

Configura dinamicamente o PYTHONPATH para permitir a importação
de módulos localizados na pasta /src e em TODAS as suas camadas
internas do ciclo de Machine Learning.

Este script deve ser executado antes de qualquer import
nos notebooks ou ambientes interativos.
"""

from pathlib import Path
import sys
import logging
from typing import Iterable


# =========================
# Logging configuration
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)


def add_paths_to_pythonpath(paths: Iterable[Path]) -> None:
    """
    Adiciona múltiplos caminhos ao PYTHONPATH, se ainda não existirem.

    Args:
        paths (Iterable[Path]): Caminhos a serem adicionados.
    """
    for path in paths:
        resolved_path = str(path.resolve())

        if resolved_path not in sys.path:
            sys.path.insert(0, resolved_path)
            logger.info("Added to PYTHONPATH: %s", resolved_path)
        else:
            logger.debug("Already in PYTHONPATH: %s", resolved_path)


def discover_src_layers(src_path: Path) -> list[Path]:
    """
    Descobre automaticamente as camadas válidas dentro de /src.

    Args:
        src_path (Path): Caminho da pasta src.

    Returns:
        list[Path]: Lista de pastas válidas a serem adicionadas ao PYTHONPATH.
    """
    if not src_path.exists():
        raise FileNotFoundError(f"src directory not found at: {src_path}")

    layers = [
        src_path / "config",
        src_path / "evaluation",
        src_path / "inference",
        src_path / "ingestion",
        src_path / "pipeline",
        src_path / "preprocessing",
        src_path / "training",
        src_path / "transformation",
        src_path / "utils",
        src_path / "persistence",
        src_path / "processing"
    ]

    existing_layers = [layer for layer in layers if layer.exists()]

    if not existing_layers:
        raise RuntimeError("No valid src layers found to add to PYTHONPATH.")

    return [src_path, *existing_layers]


def add_src_layers_to_path() -> None:
    """
    Localiza a pasta /src a partir da localização do arquivo atual
    e adiciona todas as camadas do projeto ao PYTHONPATH.
    """
    try:
        current_file = Path(__file__).resolve()

        # notebooks/_setup.py → notebooks → project root
        project_root = current_file.parent.parent
        src_path = project_root / "src"

        logger.info("Project root detected at: %s", project_root)
        logger.info("SRC path detected at: %s", src_path)

        layers_to_add = discover_src_layers(src_path)
        add_paths_to_pythonpath(layers_to_add)

        logger.info("PYTHONPATH successfully configured")

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

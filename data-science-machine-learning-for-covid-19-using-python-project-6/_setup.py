"""
_setup.py

Configura dinamicamente o PYTHONPATH para permitir
importações a partir da pasta /src quando executado
a partir da pasta /notebooks.
"""

from pathlib import Path
import sys
import logging


# =====================================
# LOGGING CONFIGURATION
# =====================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


def add_to_pythonpath(path: Path) -> None:
    """Adiciona caminho ao PYTHONPATH se ainda não existir."""
    resolved = str(path.resolve())

    if resolved not in sys.path:
        sys.path.insert(0, resolved)
        logger.info("Added to PYTHONPATH: %s", resolved)


def configure_project_paths() -> None:
    """
    Detecta automaticamente a raiz do projeto,
    mesmo quando executado dentro de /notebooks.
    """
    try:
        current_file = Path(__file__).resolve()

        # Se estiver rodando a partir de notebooks/
        if current_file.parent.name == "notebooks":
            project_root = current_file.parent.parent
        else:
            project_root = current_file.parent

        src_path = project_root / "src"

        if not src_path.exists():
            raise FileNotFoundError(f"'src' folder not found at {src_path}")

        logger.info("Project root detected: %s", project_root)
        logger.info("SRC detected at: %s", src_path)

        # Adiciona src
        add_to_pythonpath(src_path)

        # Adiciona automaticamente subpastas (ex: ingestion)
        for subfolder in src_path.iterdir():
            if subfolder.is_dir():
                add_to_pythonpath(subfolder)

        logger.info("PYTHONPATH successfully configured.")

    except Exception as e:
        logger.error("Error configuring PYTHONPATH: %s", e, exc_info=True)
        raise RuntimeError("Failed to configure project paths.") from e


# Executa automaticamente ao importar
configure_project_paths()

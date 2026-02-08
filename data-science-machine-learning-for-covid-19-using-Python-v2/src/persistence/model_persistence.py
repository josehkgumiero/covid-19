"""
model_persistence.py

Persistência de modelos treinados em disco.
"""

from pathlib import Path
from typing import Any
import pickle

from utils.logger import get_logger
from utils.exceptions import MLProjectException


logger = get_logger(__name__)


def save_model(
    model: Any,
    filename: str,
    directory: str = "models"
) -> Path:
    """
    Salva um modelo treinado em disco.

    Args:
        model (Any): Modelo treinado
        filename (str): Nome do arquivo
        directory (str): Diretório de saída

    Returns:
        Path: Caminho do arquivo salvo
    """
    try:
        logger.info("Iniciando salvamento do modelo")

        if model is None:
            raise MLProjectException("Modelo inválido para salvamento")

        model_dir = Path(directory)
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / filename

        with open(model_path, "wb") as file:
            pickle.dump(model, file)

        logger.info("Modelo salvo em %s", model_path)

        return model_path

    except Exception as error:
        logger.error(
            "Erro ao salvar o modelo",
            exc_info=True
        )
        raise MLProjectException(str(error)) from error

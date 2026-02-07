"""
model_persistence.py

Módulo responsável pela persistência de modelos de Machine Learning.

Inclui funcionalidades para:
- Salvar modelos treinados em disco
- Garantir rastreabilidade e reprodutibilidade
- Centralizar tratamento de erros e logging
"""

from pathlib import Path
import pickle
from typing import Any

from utils.logger import get_logger
from utils.exceptions import MLProjectException


logger = get_logger(__name__)


def save_model(
    model: Any,
    filename: str,
    directory: str = "models"
) -> Path:
    """
    Salva um modelo treinado em disco utilizando pickle.

    Args:
        model (Any): Modelo treinado (ex: scikit-learn estimator)
        filename (str): Nome do arquivo de saída (ex: model.sav)
        directory (str): Diretório onde o modelo será salvo

    Returns:
        Path: Caminho completo do arquivo salvo

    Raises:
        MLProjectException: Em caso de falha ao salvar o modelo
    """
    try:
        logger.info("Iniciando salvamento do modelo")

        if model is None:
            raise MLProjectException("Model object is None. Nothing to save.")

        model_dir = Path(directory)
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / filename

        logger.info("Salvando modelo em: %s", model_path)

        with open(model_path, "wb") as file:
            pickle.dump(model, file)

        logger.info("Modelo salvo com sucesso")

        return model_path

    except Exception as error:
        logger.error(
            "Erro ao salvar o modelo em disco",
            exc_info=True
        )
        raise MLProjectException(error) from error

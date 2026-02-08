"""
predictor.py

Responsável pela inferência/predição usando um modelo treinado.
"""

from typing import Any
import pandas as pd

from utils.logger import get_logger
from utils.exceptions import MLProjectException


logger = get_logger(__name__)


class Predictor:
    """
    Classe responsável por executar predições.
    """

    def __init__(self, model: Any) -> None:
        """
        Args:
            model (Any): Modelo treinado (ex: scikit-learn estimator)
        """
        self.model = model

    def predict(self, X: pd.DataFrame):
        """
        Executa predições.

        Args:
            X (pd.DataFrame): Features de entrada

        Returns:
            array-like: Predições do modelo
        """
        try:
            logger.info("Executando predições")

            if self.model is None:
                raise MLProjectException("Modelo não inicializado")

            predictions = self.model.predict(X)

            logger.info("Predições concluídas")

            return predictions

        except Exception as error:
            logger.error(
                "Erro durante a predição",
                exc_info=True
            )
            raise MLProjectException(str(error)) from error

import pandas as pd
from typing import Any
from utils.logger import get_logger
from utils.exceptions import MLProjectException

logger = get_logger(__name__)

class Predictor:
    """Responsável por gerar predições."""

    def __init__(self, model: Any):
        self.model = model

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        try:
            logger.info("Gerando previsões")
            return self.model.predict(X_test)
        except Exception as e:
            logger.error("Erro na predição", exc_info=True)
            raise MLProjectException(e)

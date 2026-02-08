"""
imbalance_handler.py

Implementa utilitários para tratamento de desbalanceamento de classes.
Inclui ClusterCentroids (under-sampling).
"""

from typing import Tuple
import pandas as pd
from imblearn.under_sampling import ClusterCentroids

from utils.logger import get_logger
from utils.exceptions import MLProjectException


logger = get_logger(__name__)


class ClusterCentroidsUnderSampler:
    """
    Under-sampling via ClusterCentroids.
    """

    def __init__(
        self,
        random_state: int = 42,
        sampling_strategy: str = "auto"
    ) -> None:
        self.random_state = random_state
        self.sampling_strategy = sampling_strategy
        self._sampler = ClusterCentroids(
            random_state=self.random_state,
            sampling_strategy=self.sampling_strategy
        )

    def fit_resample(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Aplica under-sampling.

        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Dados balanceados
        """
        try:
            logger.info("Aplicando ClusterCentroids under-sampling")

            if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
                raise MLProjectException("Tipos inválidos para X ou y")

            X_resampled, y_resampled = self._sampler.fit_resample(X, y)

            logger.info(
                "Under-sampling concluído | antes=%d | depois=%d",
                len(y),
                len(y_resampled)
            )

            return X_resampled, y_resampled

        except Exception as error:
            logger.error(
                "Erro no ClusterCentroids under-sampling",
                exc_info=True
            )
            raise MLProjectException(str(error)) from error

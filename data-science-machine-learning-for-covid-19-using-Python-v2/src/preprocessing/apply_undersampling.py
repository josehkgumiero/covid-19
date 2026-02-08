"""
apply_undersampling.py

Aplica under-sampling (ClusterCentroids) a partir de um ModelTrainer
já treinado, sem expor estado interno no notebook.
"""

from typing import Tuple
import pandas as pd

from preprocessing.imbalance_handler import ClusterCentroidsUnderSampler
from utils.logger import get_logger
from utils.exceptions import MLProjectException


logger = get_logger(__name__)


def apply_undersampling_from_trainer(
    trainer
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Aplica under-sampling usando os dados de treino do trainer.

    Args:
        trainer: Instância treinada de ModelTrainer

    Returns:
        Tuple[pd.DataFrame, pd.Series]: X e y balanceados
    """
    try:
        logger.info("Iniciando under-sampling a partir do trainer")

        if not hasattr(trainer, "X_train") or not hasattr(trainer, "y_train"):
            raise MLProjectException(
                "Trainer não contém dados de treino. "
                "Execute trainer.train() primeiro."
            )

        sampler = ClusterCentroidsUnderSampler()

        X_resampled, y_resampled = sampler.fit_resample(
            X=trainer.X_train,
            y=trainer.y_train
        )

        logger.info("Under-sampling aplicado com sucesso")

        return X_resampled, y_resampled

    except Exception as error:
        logger.error(
            "Erro ao aplicar under-sampling a partir do trainer",
            exc_info=True
        )
        raise MLProjectException(str(error)) from error

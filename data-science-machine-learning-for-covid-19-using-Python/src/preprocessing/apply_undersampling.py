"""
apply_undersampling.py

Módulo responsável por aplicar under-sampling
a partir de um ModelTrainer já treinado.

Este módulo NÃO modifica o trainer,
apenas consome seu estado interno de forma segura.
"""

from typing import Tuple

import pandas as pd

from preprocessing.imbalance_handler import ClusterCentroidsUnderSampler
from utils.exceptions import MLProjectException
from utils.logger import get_logger


logger = get_logger(__name__)


def apply_undersampling_from_trainer(
    trainer
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Aplica under-sampling utilizando os dados de treino
    armazenados no ModelTrainer.

    Args:
        trainer: Instância treinada de ModelTrainer

    Returns:
        Tuple[pd.DataFrame, pd.Series]: X e y balanceados
    """
    try:
        logger.info("Iniciando under-sampling a partir do trainer")

        if not hasattr(trainer, "X_train") or not hasattr(trainer, "y_train"):
            raise MLProjectException(
                "Trainer does not contain training data. "
                "Ensure trainer.train() was executed."
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
        raise MLProjectException(error) from error

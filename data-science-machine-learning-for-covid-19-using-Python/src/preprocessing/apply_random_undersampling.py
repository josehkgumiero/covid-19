"""
apply_random_undersampling.py

Módulo responsável por aplicar RandomUnderSampler
utilizando os dados de treino armazenados no ModelTrainer.

Este módulo:
- NÃO altera o trainer
- NÃO expõe X_train / y_train no notebook
- Segue a arquitetura atual do projeto
"""

from typing import Tuple

import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

from utils.logger import get_logger
from utils.exceptions import MLProjectException


logger = get_logger(__name__)


def apply_random_undersampling_from_trainer(
    trainer,
    random_state: int = 0
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Aplica RandomUnderSampler usando os dados de treino
    armazenados no ModelTrainer.

    Args:
        trainer: Instância treinada de ModelTrainer
        random_state (int): Semente de aleatoriedade

    Returns:
        Tuple[pd.DataFrame, pd.Series]: X e y balanceados
    """
    try:
        logger.info("Iniciando RandomUnderSampler a partir do trainer")

        if not hasattr(trainer, "X_train") or not hasattr(trainer, "y_train"):
            raise MLProjectException(
                "Trainer does not contain training data. "
                "Ensure trainer.train() was executed."
            )

        sampler = RandomUnderSampler(random_state=random_state)

        X_resampled, y_resampled = sampler.fit_resample(
            trainer.X_train,
            trainer.y_train
        )

        logger.info(
            "RandomUnderSampler aplicado | Tamanho antes: %d | depois: %d",
            len(trainer.y_train),
            len(y_resampled)
        )

        return X_resampled, y_resampled

    except Exception as error:
        logger.error(
            "Erro ao aplicar RandomUnderSampler a partir do trainer",
            exc_info=True
        )
        raise MLProjectException(error) from error

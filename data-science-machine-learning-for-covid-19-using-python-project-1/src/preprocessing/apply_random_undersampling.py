"""
apply_random_undersampling.py

Aplica RandomUnderSampler a partir de um ModelTrainer já treinado,
sem expor X_train e y_train diretamente no notebook.
"""

from typing import Tuple
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

from utils.logger import get_logger
from utils.exceptions import MLProjectException


logger = get_logger(__name__)


def apply_random_undersampling_from_trainer(
    trainer,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Aplica RandomUnderSampler usando os dados de treino do trainer.

    Args:
        trainer: Instância treinada de ModelTrainer
        random_state (int): Seed de aleatoriedade

    Returns:
        Tuple[pd.DataFrame, pd.Series]: X e y balanceados
    """
    try:
        logger.info("Iniciando RandomUnderSampler a partir do trainer")

        if not hasattr(trainer, "X_train") or not hasattr(trainer, "y_train"):
            raise MLProjectException(
                "Trainer não contém dados de treino. "
                "Execute trainer.train() primeiro."
            )

        sampler = RandomUnderSampler(random_state=random_state)

        X_resampled, y_resampled = sampler.fit_resample(
            trainer.X_train,
            trainer.y_train
        )

        logger.info(
            "RandomUnderSampler aplicado | antes=%d | depois=%d",
            len(trainer.y_train),
            len(y_resampled)
        )

        return X_resampled, y_resampled

    except Exception as error:
        logger.error(
            "Erro ao aplicar RandomUnderSampler",
            exc_info=True
        )
        raise MLProjectException(str(error)) from error

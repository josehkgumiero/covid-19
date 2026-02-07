"""
apply_smote_oversampling.py

Módulo responsável por aplicar técnicas de over-sampling
(SMOTE e ADASYN) utilizando os dados de treino armazenados
no ModelTrainer.

Este módulo:
- NÃO altera o trainer
- NÃO expõe X_train / y_train no notebook
- Segue a arquitetura atual do projeto
"""

from typing import Tuple, Literal

import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN

from utils.logger import get_logger
from utils.exceptions import MLProjectException


logger = get_logger(__name__)


def apply_smote_from_trainer(
    trainer,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Aplica SMOTE usando os dados de treino do ModelTrainer.

    Args:
        trainer: Instância treinada de ModelTrainer
        random_state (int): Semente de aleatoriedade

    Returns:
        Tuple[pd.DataFrame, pd.Series]: X e y balanceados
    """
    return _apply_oversampling(
        trainer=trainer,
        method="smote",
        random_state=random_state
    )


def apply_adasyn_from_trainer(
    trainer,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Aplica ADASYN usando os dados de treino do ModelTrainer.

    Args:
        trainer: Instância treinada de ModelTrainer
        random_state (int): Semente de aleatoriedade

    Returns:
        Tuple[pd.DataFrame, pd.Series]: X e y balanceados
    """
    return _apply_oversampling(
        trainer=trainer,
        method="adasyn",
        random_state=random_state
    )


def _apply_oversampling(
    trainer,
    method: Literal["smote", "adasyn"],
    random_state: int
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Função interna compartilhada para over-sampling.
    """
    try:
        logger.info("Iniciando over-sampling (%s) a partir do trainer", method)

        if not hasattr(trainer, "X_train") or not hasattr(trainer, "y_train"):
            raise MLProjectException(
                "Trainer does not contain training data. "
                "Ensure trainer.train() was executed."
            )

        if method == "smote":
            sampler = SMOTE(random_state=random_state)
        elif method == "adasyn":
            sampler = ADASYN(random_state=random_state)
        else:
            raise MLProjectException(f"Unsupported oversampling method: {method}")

        X_resampled, y_resampled = sampler.fit_resample(
            trainer.X_train,
            trainer.y_train
        )

        logger.info(
            "%s aplicado | Tamanho antes: %d | depois: %d",
            method.upper(),
            len(trainer.y_train),
            len(y_resampled)
        )

        return X_resampled, y_resampled

    except Exception as error:
        logger.error(
            "Erro ao aplicar %s oversampling a partir do trainer",
            method.upper(),
            exc_info=True
        )
        raise MLProjectException(error) from error
    
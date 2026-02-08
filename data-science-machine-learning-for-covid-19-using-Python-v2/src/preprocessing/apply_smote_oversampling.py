"""
apply_smote_oversampling.py

Aplica técnicas de over-sampling (SMOTE e ADASYN)
a partir de um ModelTrainer treinado.
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
    Aplica SMOTE usando os dados de treino do trainer.
    """
    return _apply_oversampling(trainer, "smote", random_state)


def apply_adasyn_from_trainer(
    trainer,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Aplica ADASYN usando os dados de treino do trainer.
    """
    return _apply_oversampling(trainer, "adasyn", random_state)


def _apply_oversampling(
    trainer,
    method: Literal["smote", "adasyn"],
    random_state: int
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Função interna compartilhada para over-sampling.
    """
    try:
        logger.info("Iniciando over-sampling (%s)", method.upper())

        if not hasattr(trainer, "X_train") or not hasattr(trainer, "y_train"):
            raise MLProjectException(
                "Trainer não contém dados de treino. "
                "Execute trainer.train() primeiro."
            )

        if method == "smote":
            sampler = SMOTE(random_state=random_state)
        elif method == "adasyn":
            sampler = ADASYN(random_state=random_state)
        else:
            raise MLProjectException(f"Método não suportado: {method}")

        X_resampled, y_resampled = sampler.fit_resample(
            trainer.X_train,
            trainer.y_train
        )

        logger.info(
            "%s aplicado | antes=%d | depois=%d",
            method.upper(),
            len(trainer.y_train),
            len(y_resampled)
        )

        return X_resampled, y_resampled

    except Exception as error:
        logger.error(
            "Erro ao aplicar over-sampling",
            exc_info=True
        )
        raise MLProjectException(str(error)) from error

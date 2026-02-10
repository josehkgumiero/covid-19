from typing import Tuple
import numpy as np
from tensorflow.keras.models import Model

from src.utils.logger import get_logger


class Trainer:
    """
    Responsável por treinar e avaliar modelos Keras.
    """

    def __init__(
        self,
        model: Model,
        batch_size: int = 64,
        epochs: int = 25,
    ) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ):
        """
        Treina o modelo com validação.
        """
        self.logger.info("Iniciando treinamento do modelo")

        history = self.model.fit(
            x_train,
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(x_val, y_val),
        )

        self.logger.info("Treinamento concluído")
        return history

from typing import Tuple
import numpy as np
from tensorflow.keras.models import Model

from src.utils.logger import get_logger


class Evaluator:
    """
    Classe responsável pela avaliação de modelos Keras.
    """

    def __init__(self, model: Model, batch_size: int = 64) -> None:
        """
        Inicializa o avaliador.

        Args:
            model (Model): Modelo Keras treinado.
            batch_size (int): Tamanho do batch para avaliação.
        """
        self.logger = get_logger(self.__class__.__name__)
        self.model = model
        self.batch_size = batch_size

    def evaluate(
        self,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Avalia o modelo em dados de teste.

        Args:
            x_test (np.ndarray): Dados de entrada de teste.
            y_test (np.ndarray): Labels de teste.

        Returns:
            Tuple[float, float]: (loss, accuracy)
        """
        try:
            self.logger.info("Iniciando avaliação do modelo")

            loss, accuracy = self.model.evaluate(
                x_test,
                y_test,
                batch_size=self.batch_size,
                verbose=0,
            )

            self.logger.info(
                "Avaliação concluída (loss=%.4f, accuracy=%.4f)",
                loss,
                accuracy,
            )

            return loss, accuracy

        except Exception as e:
            self.logger.error("Erro durante a avaliação do modelo", exc_info=True)
            raise RuntimeError("Falha na avaliação do modelo") from e

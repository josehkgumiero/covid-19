from typing import Dict
import matplotlib.pyplot as plt

from src.utils.logger import get_logger


class TrainingPlotter:
    """
    Responsável por visualizar métricas de treinamento Keras.
    """

    def __init__(self, history: Dict) -> None:
        """
        Inicializa o plotter.

        Args:
            history (Dict): history.history do Keras.
        """
        self.logger = get_logger(self.__class__.__name__)
        self.history = history

    def plot_accuracy(self) -> None:
        """
        Plota a acurácia de treino e validação.
        """
        try:
            self.logger.info("Plotando acurácia")

            plt.plot(self.history["accuracy"])
            plt.plot(self.history["val_accuracy"])
            plt.title("Model accuracy")
            plt.ylabel("Accuracy")
            plt.xlabel("Epoch")
            plt.legend(["Train", "Validation"], loc="upper left")
            plt.show()

        except KeyError as e:
            self.logger.error("Chaves de acurácia não encontradas", exc_info=True)
            raise RuntimeError("Histórico não contém métricas de acurácia") from e

    def plot_loss(self) -> None:
        """
        Plota a loss de treino e validação.
        """
        try:
            self.logger.info("Plotando loss")

            plt.plot(self.history["loss"])
            plt.plot(self.history["val_loss"])
            plt.title("Model loss")
            plt.ylabel("Loss")
            plt.xlabel("Epoch")
            plt.legend(["Train", "Validation"], loc="upper left")
            plt.show()

        except KeyError as e:
            self.logger.error("Chaves de loss não encontradas", exc_info=True)
            raise RuntimeError("Histórico não contém métricas de loss") from e

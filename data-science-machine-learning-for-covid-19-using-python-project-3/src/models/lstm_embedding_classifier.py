from typing import Optional

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.initializers import Constant

from src.utils.logger import get_logger


class LSTMEmbeddingClassifier:
    """
    Classificador LSTM com camada de embedding pré-treinada.
    """

    def __init__(
        self,
        num_words: int,
        embedding_dim: int,
        embedding_matrix: np.ndarray,
        max_length: int,
        lstm_units: int = 32,
        dropout: float = 0.2,
        trainable_embedding: bool = False,
    ) -> None:
        """
        Inicializa o classificador.

        Args:
            num_words (int): Tamanho do vocabulário.
            embedding_dim (int): Dimensão dos embeddings.
            embedding_matrix (np.ndarray): Matriz de embeddings.
            max_length (int): Tamanho máximo da sequência.
            lstm_units (int): Número de unidades LSTM.
            dropout (float): Taxa de dropout.
            trainable_embedding (bool): Se a embedding é treinável.
        """
        self.logger = get_logger(self.__class__.__name__)

        self.num_words = num_words
        self.embedding_dim = embedding_dim
        self.embedding_matrix = embedding_matrix
        self.max_length = max_length
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.trainable_embedding = trainable_embedding

        self.model: Optional[Sequential] = None

        self.logger.info(
            "LSTMEmbeddingClassifier inicializado "
            "(num_words=%d, embedding_dim=%d, max_length=%d)",
            num_words,
            embedding_dim,
            max_length,
        )

    def build(self) -> Sequential:
        """
        Constrói e compila o modelo Keras.

        Returns:
            Sequential: Modelo compilado.
        """
        try:
            self.logger.info("Construindo modelo LSTM")

            model = Sequential()

            model.add(
                Embedding(
                    input_dim=self.num_words,
                    output_dim=self.embedding_dim,
                    embeddings_initializer=Constant(self.embedding_matrix),
                    input_length=self.max_length,
                    trainable=self.trainable_embedding,
                )
            )

            model.add(
                LSTM(
                    units=self.lstm_units,
                    dropout=self.dropout,
                    recurrent_dropout=self.dropout,
                )
            )

            model.add(Dense(1, activation="sigmoid"))

            model.compile(
                optimizer="adam",
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )

            self.model = model

            self.logger.info("Modelo LSTM construído e compilado com sucesso")

            return model

        except Exception as e:
            self.logger.error("Erro ao construir modelo LSTM", exc_info=True)
            raise RuntimeError("Falha na construção do modelo LSTM") from e

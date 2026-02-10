from typing import List, Tuple
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.utils.logger import get_logger


class SequenceBuilder:
    """
    Constrói sequências com padding e split treino/teste
    a partir de textos ou tokens.
    """

    def __init__(
        self,
        max_length: int = 100,
        test_split: float = 0.2,
        random_state: int = 42,
    ) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.max_length = max_length
        self.test_split = test_split
        self.random_state = random_state

        # Tokenizer ÚNICO e controlado
        self.tokenizer = Tokenizer()

        self.logger.info(
            "SequenceBuilder inicializado "
            "(max_length=%d, test_split=%.2f)",
            max_length,
            test_split,
        )

    def build(
        self,
        texts: List[str] | List[List[str]],
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Tokeniza textos, aplica padding e realiza split treino/teste.
        """
        try:
            if len(texts) != len(labels):
                raise ValueError("Texts e labels têm tamanhos diferentes")

            self.logger.info("Tokenizando textos")
            self.tokenizer.fit_on_texts(texts)

            sequences = self.tokenizer.texts_to_sequences(texts)
            padded = pad_sequences(sequences, maxlen=self.max_length)

            # Shuffle reproduzível
            np.random.seed(self.random_state)
            indices = np.arange(padded.shape[0])
            np.random.shuffle(indices)

            padded = padded[indices]
            labels = labels[indices]

            test_size = int(self.test_split * padded.shape[0])

            x_train = padded[:-test_size]
            y_train = labels[:-test_size]

            x_test = padded[-test_size:]
            y_test = labels[-test_size:]

            self.logger.info(
                "Split concluído (train=%d, test=%d)",
                x_train.shape[0],
                x_test.shape[0],
            )

            return x_train, y_train, x_test, y_test

        except Exception as e:
            self.logger.error("Erro ao construir sequências", exc_info=True)
            raise RuntimeError("Falha na construção das sequências") from e

    def get_tokenizer(self) -> Tokenizer:
        """
        Retorna o tokenizer utilizado internamente.
        """
        return self.tokenizer

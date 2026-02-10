import numpy as np
from typing import Dict

from src.utils.logger import get_logger


class EmbeddingMatrixBuilder:
    """
    Constrói a matriz de embeddings alinhada ao tokenizer oficial.
    """

    def __init__(self, embedding_dim: int) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.embedding_dim = embedding_dim

    def build(
        self,
        word_index: Dict[str, int],
        embedding_index: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Cria a matriz de embeddings.

        Args:
            word_index: tokenizer.word_index (OFICIAL)
            embedding_index: dicionário {palavra: vetor}

        Returns:
            np.ndarray: embedding_matrix
        """
        if not word_index:
            raise ValueError("word_index vazio")

        if not embedding_index:
            raise ValueError("embedding_index vazio")

        num_words = len(word_index) + 1
        embedding_matrix = np.zeros((num_words, self.embedding_dim))

        hits = 0
        misses = 0

        for word, idx in word_index.items():
            vector = embedding_index.get(word)
            if vector is not None:
                embedding_matrix[idx] = vector
                hits += 1
            else:
                misses += 1

        self.logger.info(
            "Embedding matrix criada (hits=%d, misses=%d)",
            hits,
            misses,
        )

        return embedding_matrix

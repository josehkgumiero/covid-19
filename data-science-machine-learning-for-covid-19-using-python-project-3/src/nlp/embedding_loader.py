from pathlib import Path
from typing import Dict

import numpy as np

from src.utils.logger import get_logger


class EmbeddingLoader:
    """
    Classe responsável por carregar embeddings salvos no formato Word2Vec (texto).

    Cada linha do arquivo deve seguir o formato:
        palavra valor1 valor2 valor3 ...
    """

    def __init__(self, embedding_path: str | Path) -> None:
        """
        Inicializa o loader de embeddings.

        Args:
            embedding_path (str | Path): Caminho para o arquivo de embeddings.
        """
        self.logger = get_logger(self.__class__.__name__)
        self.embedding_path = Path(embedding_path)

        self.logger.info("EmbeddingLoader inicializado: %s", self.embedding_path)

    def load(self) -> Dict[str, np.ndarray]:
        """
        Carrega os embeddings do arquivo para um dicionário.

        Returns:
            Dict[str, np.ndarray]: Dicionário {palavra: vetor}.

        Raises:
            FileNotFoundError: Se o arquivo não existir.
            RuntimeError: Se ocorrer erro de leitura ou parsing.
        """
        try:
            if not self.embedding_path.exists():
                raise FileNotFoundError(
                    f"Arquivo de embeddings não encontrado: {self.embedding_path}"
                )

            embeddings: Dict[str, np.ndarray] = {}

            self.logger.info("Iniciando leitura do arquivo de embeddings")

            # Uso de context manager garante fechamento do arquivo
            with self.embedding_path.open(encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    values = line.strip().split()

                    # Linha vazia ou inválida
                    if len(values) < 2:
                        self.logger.warning(
                            "Linha %d ignorada (formato inválido)", line_number
                        )
                        continue

                    word = values[0]

                    try:
                        vector = np.asarray(values[1:], dtype="float32")
                    except ValueError:
                        self.logger.warning(
                            "Linha %d ignorada (vetor inválido)", line_number
                        )
                        continue

                    embeddings[word] = vector

            self.logger.info(
                "Embeddings carregados com sucesso (%d palavras)", len(embeddings)
            )

            return embeddings

        except Exception as e:
            self.logger.error("Erro ao carregar embeddings", exc_info=True)
            raise RuntimeError("Falha ao carregar embeddings Word2Vec") from e

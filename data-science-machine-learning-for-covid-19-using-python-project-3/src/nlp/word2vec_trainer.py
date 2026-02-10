from pathlib import Path
from typing import List

from gensim.models import Word2Vec

from src.utils.logger import get_logger


class Word2VecTrainer:
    """
    Classe responsável por treinar e persistir modelos Word2Vec.

    Responsabilidades:
        - Treinar embeddings Word2Vec
        - Gerenciar hiperparâmetros
        - Salvar embeddings em formato texto (Word2Vec)
    """

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 1,
        workers: int = 4,
    ) -> None:
        """
        Inicializa o treinador de Word2Vec.

        Args:
            vector_size (int): Dimensão do embedding.
            window (int): Tamanho da janela de contexto.
            min_count (int): Frequência mínima de palavras.
            workers (int): Número de threads.
        """
        self.logger = get_logger(self.__class__.__name__)

        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers

        self.model: Word2Vec | None = None

        self.logger.info(
            "Word2VecTrainer inicializado "
            "(vector_size=%d, window=%d, min_count=%d, workers=%d)",
            vector_size,
            window,
            min_count,
            workers,
        )

    def train(self, sentences: List[List[str]]) -> Word2Vec:
        """
        Treina o modelo Word2Vec a partir de sentenças tokenizadas.

        Args:
            sentences (List[List[str]]): Lista de listas de tokens.

        Returns:
            Word2Vec: Modelo Word2Vec treinado.
        """
        try:
            if not sentences:
                raise ValueError("Lista de sentenças vazia")

            self.logger.info("Iniciando treinamento do Word2Vec")

            self.model = Word2Vec(
                sentences=sentences,
                vector_size=self.vector_size,
                window=self.window,
                min_count=self.min_count,
                workers=self.workers,
            )

            self.logger.info(
                "Treinamento concluído (vocabulário=%d palavras)",
                len(self.model.wv),
            )

            return self.model

        except Exception as e:
            self.logger.error("Erro durante o treinamento do Word2Vec", exc_info=True)
            raise RuntimeError("Falha no treinamento do Word2Vec") from e

    def save_embeddings(self, output_path: str | Path, binary: bool = False) -> None:
        """
        Salva os embeddings treinados no formato Word2Vec.

        Args:
            output_path (str | Path): Caminho do arquivo de saída.
            binary (bool): Se True, salva em formato binário.
        """
        try:
            if self.model is None:
                raise RuntimeError("Modelo Word2Vec ainda não foi treinado")

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            self.logger.info("Salvando embeddings em %s", output_path)

            self.model.wv.save_word2vec_format(
                fname=str(output_path),
                binary=binary,
            )

            self.logger.info("Embeddings salvos com sucesso")

        except Exception as e:
            self.logger.error("Erro ao salvar embeddings", exc_info=True)
            raise RuntimeError("Falha ao salvar embeddings Word2Vec") from e
    
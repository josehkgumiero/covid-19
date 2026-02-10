import string
from typing import List

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from src.utils.logger import get_logger


class TextPreprocessor:
    """
    Classe responsável pelo pré-processamento de textos para NLP.

    Etapas:
        - Tokenização
        - Lowercase
        - Remoção de pontuação
        - Remoção de tokens não alfabéticos
        - Remoção de stopwords
    """

    def __init__(self, language: str = "english") -> None:
        """
        Inicializa o pré-processador.

        Args:
            language (str): Idioma para stopwords (padrão: 'english').
        """
        self.logger = get_logger(self.__class__.__name__)
        self.language = language

        # Pré-carrega stopwords para evitar custo repetido
        self.stop_words = set(stopwords.words(self.language))

        # Tabela de tradução para remoção de pontuação
        self._punctuation_table = str.maketrans("", "", string.punctuation)

        self.logger.info("TextPreprocessor inicializado (language=%s)", self.language)

    def preprocess_text(self, text: str) -> List[str]:
        """
        Aplica o pré-processamento completo a um único texto.

        Args:
            text (str): Texto bruto de entrada.

        Returns:
            List[str]: Lista de tokens limpos.
        """
        try:
            # Tokenização
            tokens = word_tokenize(text)

            # Normalização (lowercase)
            tokens = [token.lower() for token in tokens]

            # Remoção de pontuação
            tokens = [token.translate(self._punctuation_table) for token in tokens]

            # Mantém apenas tokens alfabéticos
            tokens = [token for token in tokens if token.isalpha()]

            # Remove stopwords
            tokens = [token for token in tokens if token not in self.stop_words]

            return tokens

        except Exception as e:
            self.logger.error(
                "Erro ao processar texto: %s", text[:50], exc_info=True
            )
            raise RuntimeError("Falha no pré-processamento de texto") from e

    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str) -> List[List[str]]:
        """
        Aplica o pré-processamento a uma coluna de texto de um DataFrame.

        Args:
            df (pd.DataFrame): DataFrame contendo os textos.
            text_column (str): Nome da coluna de texto.

        Returns:
            List[List[str]]: Lista de listas de tokens processados.
        """
        try:
            if text_column not in df.columns:
                raise ValueError(f"Coluna '{text_column}' não encontrada no DataFrame")

            self.logger.info("Iniciando pré-processamento da coluna '%s'", text_column)

            texts = df[text_column].astype(str).tolist()

            processed_texts = [
                self.preprocess_text(text) for text in texts
            ]

            self.logger.info(
                "Pré-processamento concluído (%d registros)", len(processed_texts)
            )

            return processed_texts

        except Exception as e:
            self.logger.error("Erro ao processar DataFrame", exc_info=True)
            raise RuntimeError("Falha no pré-processamento do DataFrame") from e

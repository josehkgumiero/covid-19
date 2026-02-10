from typing import Tuple
import pandas as pd
import numpy as np

from src.utils.logger import get_logger


class DatasetSplitter:
    """
    Classe responsável por realizar a separação de datasets
    em conjuntos de treino e teste.
    """

    def __init__(
        self,
        text_column: str = "text",
        target_column: str = "target",
        train_end_index: int | None = None,
        train_ratio: float | None = None,
    ) -> None:
        """
        Inicializa o splitter.

        Args:
            text_column (str): Nome da coluna de texto.
            target_column (str): Nome da coluna target.
            train_end_index (int | None): Índice final do conjunto de treino.
            train_ratio (float | None): Proporção de treino (ex: 0.8).
        """
        self.logger = get_logger(self.__class__.__name__)
        self.text_column = text_column
        self.target_column = target_column
        self.train_end_index = train_end_index
        self.train_ratio = train_ratio

        if train_end_index is None and train_ratio is None:
            raise ValueError(
                "Informe train_end_index OU train_ratio para o split"
            )

        self.logger.info(
            "DatasetSplitter inicializado "
            "(text=%s, target=%s, train_end_index=%s, train_ratio=%s)",
            text_column,
            target_column,
            train_end_index,
            train_ratio,
        )

    def split(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Realiza o split do DataFrame em treino e teste.

        Args:
            df (pd.DataFrame): DataFrame de entrada.

        Returns:
            Tuple contendo:
                x_train, y_train, x_test, y_test
        """
        try:
            self._validate_dataframe(df)

            if self.train_ratio is not None:
                train_size = int(len(df) * self.train_ratio)
            else:
                train_size = self.train_end_index + 1

            self.logger.info("Separando dataset (train_size=%d)", train_size)

            x_train = df.loc[: train_size - 1, self.text_column].values
            y_train = df.loc[: train_size - 1, self.target_column].values

            x_test = df.loc[train_size:, self.text_column].values
            y_test = df.loc[train_size:, self.target_column].values

            self.logger.info(
                "Split concluído (train=%d, test=%d)",
                len(x_train),
                len(x_test),
            )

            return x_train, y_train, x_test, y_test

        except Exception as e:
            self.logger.error("Erro ao realizar split do dataset", exc_info=True)
            raise RuntimeError("Falha ao separar dataset") from e

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """
        Valida o DataFrame antes do split.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("O objeto fornecido não é um pandas.DataFrame")

        required_columns = {self.text_column, self.target_column}
        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(f"Colunas obrigatórias ausentes: {missing}")

        if df.empty:
            raise ValueError("DataFrame vazio")

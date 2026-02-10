from typing import List
import pandas as pd

from src.utils.logger import get_logger


class DataFrameSchema:
    """
    Classe responsável por validar e aplicar um schema
    mínimo a um pandas DataFrame.
    """

    def __init__(self, required_columns: List[str]) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.required_columns = required_columns

    def validate(self, df: pd.DataFrame) -> None:
        """
        Valida se o DataFrame contém todas as colunas obrigatórias.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("O objeto fornecido não é um pandas.DataFrame")

        missing = set(self.required_columns) - set(df.columns)
        if missing:
            self.logger.error("Colunas obrigatórias ausentes: %s", missing)
            raise ValueError(f"Colunas obrigatórias ausentes: {missing}")

        self.logger.info("Validação de schema concluída com sucesso")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica o schema ao DataFrame retornando apenas as colunas exigidas.
        """
        self.validate(df)

        self.logger.info("Aplicando schema com colunas: %s", self.required_columns)
        return df.loc[:, self.required_columns].copy()

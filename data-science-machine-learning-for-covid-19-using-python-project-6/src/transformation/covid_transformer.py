"""
covid_transformer.py

Camada de TRANSFORMAÇÃO.
Responsável por regras de negócio e filtros.
"""

import logging
import pandas as pd


logger = logging.getLogger(__name__)


class CovidDataTransformer:
    """
    Classe responsável por aplicar transformações
    no dataset de COVID-19.
    """

    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Entrada deve ser um pandas DataFrame.")
        self._df = df.copy()

    def filter_by_country(self, country: str) -> pd.DataFrame:
        """
        Filtra o dataset por país.

        :param country: Nome do país (ex: 'India')
        :return: DataFrame filtrado
        """
        if "Country/Region" not in self._df.columns:
            raise KeyError("Coluna 'Country/Region' não encontrada.")

        filtered_df = self._df[self._df["Country/Region"] == country]

        if filtered_df.empty:
            raise ValueError(f"Nenhum dado encontrado para: {country}")

        logger.info("Filtro aplicado para país: %s", country)

        return filtered_df.copy()

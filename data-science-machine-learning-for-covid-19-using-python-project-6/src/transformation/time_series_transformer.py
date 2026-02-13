"""
time_series_transformer.py

Responsável por transformar dados wide-format
em série temporal (ds, y) compatível com Prophet.
"""

import pandas as pd
import logging


logger = logging.getLogger(__name__)


class TimeSeriesTransformer:
    """
    Converte dataset de COVID em formato
    adequado para modelagem de séries temporais.
    """

    METADATA_COLUMNS = [
        "Province/State",
        "Country/Region",
        "Lat",
        "Long"
    ]

    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Entrada deve ser um pandas DataFrame.")
        self.df = df.copy()

    # =====================================================
    # WIDE → LONG (Prophet Format)
    # =====================================================
    def to_prophet_format(self) -> pd.DataFrame:
        """
        Converte o dataset para formato (ds, y).

        :return: DataFrame com colunas ds (data) e y (valor acumulado)
        """

        missing_cols = [
            col for col in self.METADATA_COLUMNS
            if col not in self.df.columns
        ]

        if missing_cols:
            raise KeyError(f"Colunas esperadas não encontradas: {missing_cols}")

        try:
            logger.info("Convertendo dataset para formato long (ds, y)...")

            df_long = self.df.drop(columns=self.METADATA_COLUMNS)
            df_long = df_long.T.reset_index()
            df_long.columns = ["ds", "y"]
            df_long["ds"] = pd.to_datetime(df_long["ds"])

            logger.info("Conversão concluída com sucesso.")
            return df_long

        except Exception as e:
            logger.error("Erro na transformação para formato Prophet.", exc_info=True)
            raise RuntimeError("Falha ao transformar dataset.") from e

    # =====================================================
    # CUMULATIVE → DAILY
    # =====================================================
    @staticmethod
    def to_daily_cases(df: pd.DataFrame) -> pd.DataFrame:
        """
        Converte série acumulada (cumulative cases)
        em casos diários.

        - Aplica diff()
        - Preenche NaN com 0
        - Remove valores negativos (correções históricas)

        :param df: DataFrame no formato (ds, y)
        :return: DataFrame com valores diários
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Entrada deve ser um pandas DataFrame.")

        if "y" not in df.columns:
            raise KeyError("Coluna 'y' não encontrada no DataFrame.")

        try:
            logger.info("Convertendo série acumulada para casos diários...")

            df_daily = df.copy()

            # Calcula diferença diária
            df_daily["y"] = df_daily["y"].diff()

            # Preenche primeiro valor com 0 (em vez de remover)
            df_daily["y"] = df_daily["y"].fillna(0)

            # Remove valores negativos causados por revisões históricas
            df_daily["y"] = df_daily["y"].clip(lower=0)

            logger.info("Conversão para casos diários concluída.")

            return df_daily

        except Exception as e:
            logger.error("Erro ao converter para casos diários.", exc_info=True)
            raise RuntimeError("Falha na transformação para casos diários.") from e

    # =====================================================
    # FILTER BY DATE
    # =====================================================
    @staticmethod
    def filter_until_date(df: pd.DataFrame, end_date: str) -> pd.DataFrame:
        """
        Filtra a série temporal até uma data específica.

        :param df: DataFrame com coluna 'ds'
        :param end_date: Data limite (ex: '2020-10-31')
        :return: DataFrame filtrado
        """

        if "ds" not in df.columns:
            raise KeyError("Coluna 'ds' não encontrada no DataFrame.")

        try:
            logger.info("Filtrando dados até %s", end_date)

            end_date = pd.to_datetime(end_date)

            df_filtered = df[df["ds"] <= end_date].copy()

            logger.info("Filtro por data aplicado com sucesso.")

            return df_filtered

        except Exception as e:
            logger.error("Erro ao filtrar por data.", exc_info=True)
            raise RuntimeError("Falha ao aplicar filtro por data.") from e

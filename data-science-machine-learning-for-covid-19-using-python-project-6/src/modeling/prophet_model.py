"""
prophet_model.py

Camada de modelagem utilizando Prophet.
Responsável por treino e configuração do modelo.
"""

from prophet import Prophet
import pandas as pd
import logging


logger = logging.getLogger(__name__)


class ProphetModel:
    """
    Classe responsável por configurar e treinar
    modelo Prophet.
    """

    def __init__(
        self,
        country_holidays: str | None = None,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False
    ):
        """
        Inicializa o modelo Prophet com configurações padrão.

        :param country_holidays: Código do país (ex: 'IN')
        """

        self.model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality
        )

        if country_holidays:
            self.model.add_country_holidays(country_name=country_holidays)
            logger.info("Feriados adicionados para o país: %s", country_holidays)

        logger.info("Modelo Prophet inicializado.")

    # =====================================================
    # TRAIN
    # =====================================================
    def fit(self, df: pd.DataFrame) -> None:
        """
        Treina o modelo.

        :param df: DataFrame no formato (ds, y)
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Entrada deve ser um pandas DataFrame.")

        required_cols = {"ds", "y"}
        if not required_cols.issubset(df.columns):
            raise KeyError("DataFrame deve conter colunas 'ds' e 'y'.")

        try:
            logger.info("Iniciando treinamento do modelo...")
            self.model.fit(df)
            logger.info("Treinamento concluído com sucesso.")

        except Exception as e:
            logger.error("Erro durante o treinamento.", exc_info=True)
            raise RuntimeError("Falha no treinamento do modelo.") from e

    # =====================================================
    # FORECAST
    # =====================================================
    def predict(self, periods: int = 30) -> pd.DataFrame:
        """
        Gera previsão futura.

        :param periods: Número de dias futuros
        :return: DataFrame com forecast
        """

        try:
            logger.info("Gerando previsão para %s dias...", periods)

            future = self.model.make_future_dataframe(periods=periods)
            forecast = self.model.predict(future)

            logger.info("Previsão gerada com sucesso.")

            return forecast

        except Exception as e:
            logger.error("Erro ao gerar previsão.", exc_info=True)
            raise RuntimeError("Falha ao gerar previsão.") from e

    # =====================================================
    # TRAIN TEST SPLIT (TEMPORAL)
    # =====================================================
    @staticmethod
    def temporal_split(df: pd.DataFrame, split_index: int):
        """
        Divide série temporal em treino e teste
        baseado em índice temporal.

        :param df: DataFrame (ds, y)
        :param split_index: índice de corte
        :return: (train_df, test_df)
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Entrada deve ser um pandas DataFrame.")

        if split_index <= 0 or split_index >= len(df):
            raise ValueError("split_index inválido.")

        train = df.iloc[:split_index].copy()
        test = df.iloc[split_index:].copy()

        return train, test


    # =====================================================
    # PREDICT ON EXISTING DATA
    # =====================================================
    def predict_on_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Realiza previsão para um DataFrame já existente.

        :param df: DataFrame contendo coluna 'ds'
        :return: DataFrame com previsões
        """

        if "ds" not in df.columns:
            raise KeyError("DataFrame deve conter coluna 'ds'.")

        try:
            logger.info("Gerando previsão sobre dataset fornecido...")
            forecast = self.model.predict(df)
            logger.info("Previsão concluída.")
            return forecast

        except Exception as e:
            logger.error("Erro ao prever dados existentes.", exc_info=True)
            raise RuntimeError("Falha ao gerar previsão.") from e

    # =====================================================
    # FORECAST FUTURE
    # =====================================================
    def forecast_future(self, periods: int = 30) -> pd.DataFrame:
        """
        Gera previsão futura baseada no modelo treinado.

        :param periods: número de dias futuros
        :return: DataFrame com forecast
        """

        try:
            logger.info("Criando dataframe futuro para %s períodos...", periods)

            future = self.model.make_future_dataframe(periods=periods)
            forecast = self.model.predict(future)

            logger.info("Forecast futuro gerado com sucesso.")

            return forecast

        except Exception as e:
            logger.error("Erro ao gerar forecast futuro.", exc_info=True)
            raise RuntimeError("Falha ao gerar forecast.") from e


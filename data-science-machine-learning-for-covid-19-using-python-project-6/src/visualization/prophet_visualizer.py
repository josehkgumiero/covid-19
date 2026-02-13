"""
prophet_visualizer.py

Responsável por visualizações do modelo Prophet.
"""

import logging
import pandas as pd
from prophet.plot import plot_plotly, plot_components_plotly
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


class ProphetVisualizer:

    @staticmethod
    def plot_forecast(model, forecast: pd.DataFrame):
        """
        Plota forecast usando matplotlib (padrão Prophet).
        """
        try:
            logger.info("Gerando gráfico de forecast...")
            fig = model.model.plot(forecast)
            plt.show()

        except Exception as e:
            logger.error("Erro ao plotar forecast.", exc_info=True)
            raise RuntimeError("Falha ao gerar gráfico de forecast.") from e

    @staticmethod
    def plot_forecast_interactive(model, forecast: pd.DataFrame):
        """
        Plota forecast interativo usando Plotly.
        """
        try:
            logger.info("Gerando gráfico interativo...")
            fig = plot_plotly(model.model, forecast)
            fig.show()
            return fig

        except Exception as e:
            logger.error("Erro ao gerar gráfico interativo.", exc_info=True)
            raise RuntimeError("Falha ao gerar gráfico interativo.") from e


    # =====================================================
    # INTERACTIVE FORECAST
    # =====================================================
    @staticmethod
    def plot_forecast_interactive(model, forecast: pd.DataFrame):
        """
        Plota previsão interativa usando Plotly.
        """

        try:
            logger.info("Gerando gráfico interativo de forecast...")
            fig = plot_plotly(model.model, forecast)
            fig.show()
        except Exception as e:
            logger.error("Erro ao gerar gráfico interativo.", exc_info=True)
            raise RuntimeError("Falha ao gerar gráfico interativo.") from e


    # =====================================================
    # INTERACTIVE COMPONENTS
    # =====================================================
    @staticmethod
    def plot_components_interactive(model, forecast: pd.DataFrame):
        """
        Plota componentes do modelo (trend, seasonality, holidays).
        """

        try:
            logger.info("Gerando gráfico interativo de componentes...")
            fig = plot_components_plotly(model.model, forecast)
            fig.show()
        except Exception as e:
            logger.error("Erro ao gerar componentes interativos.", exc_info=True)
            raise RuntimeError("Falha ao gerar componentes.") from e

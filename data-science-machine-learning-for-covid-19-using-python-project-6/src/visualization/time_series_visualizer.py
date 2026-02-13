"""
time_series_visualizer.py

Responsável pela visualização de séries temporais.
"""

import matplotlib.pyplot as plt
import pandas as pd
import logging


logger = logging.getLogger(__name__)


class TimeSeriesVisualizer:
    """
    Classe responsável por visualização
    de séries temporais.
    """

    @staticmethod
    def plot_series(
        df: pd.DataFrame,
        x_col: str = "ds",
        y_col: str = "y",
        style: str = "k.",
        figsize: tuple = (10, 6),
        title: str | None = None
    ) -> None:
        """
        Plota série temporal.

        :param df: DataFrame contendo dados
        :param x_col: Nome da coluna do eixo X
        :param y_col: Nome da coluna do eixo Y
        :param style: Estilo do gráfico
        :param figsize: Tamanho da figura
        :param title: Título opcional
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Entrada deve ser um pandas DataFrame.")

        if x_col not in df.columns or y_col not in df.columns:
            raise KeyError(f"Colunas '{x_col}' e/ou '{y_col}' não encontradas.")

        try:
            logger.info("Gerando gráfico da série temporal...")

            plt.figure(figsize=figsize)
            plt.plot(df[x_col], df[y_col], style)

            plt.xlabel(x_col)
            plt.ylabel(y_col)

            if title:
                plt.title(title)

            plt.grid(True)
            plt.tight_layout()
            plt.show()

            logger.info("Gráfico gerado com sucesso.")

        except Exception as e:
            logger.error("Erro ao gerar gráfico.", exc_info=True)
            raise RuntimeError("Falha ao gerar visualização.") from e

"""
covid_data_loader.py

Camada de INGESTÃO.
Responsável apenas por carregar dados externos.
"""

from typing import Optional
import logging
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


class CovidDataLoader:
    """
    Classe responsável por carregar datasets
    de COVID-19 a partir de uma URL.
    """

    def __init__(self, url: str):
        self.url = url
        self._data: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        """
        Faz download e leitura do CSV.

        :return: DataFrame completo
        """
        try:
            logger.info("Carregando dados da URL...")
            self._data = pd.read_csv(self.url)
            logger.info("Dados carregados com sucesso.")
            return self._data

        except Exception as e:
            logger.error("Erro ao carregar dados.", exc_info=True)
            raise ConnectionError("Falha ao acessar ou ler o CSV.") from e

    def get_data(self) -> pd.DataFrame:
        """
        Retorna o DataFrame carregado.
        """
        if self._data is None:
            raise ValueError("Dados ainda não carregados. Execute load() primeiro.")
        return self._data.copy()

"""
data_loader.py

Responsável pela carga de dados do projeto.
Resolve caminhos de forma robusta a partir da raiz do projeto.
"""

from pathlib import Path
import pandas as pd

from utils.logger import get_logger
from utils.exceptions import MLProjectException


logger = get_logger(__name__)


class DataLoader:
    """
    Classe para carregamento de dados.
    """

    def __init__(self, data_dir_name: str = "data") -> None:
        """
        Inicializa o loader resolvendo a raiz do projeto.

        Args:
            data_dir_name (str): Nome do diretório de dados (default: data)
        """
        self.project_root = self._find_project_root()
        self.data_root = self.project_root / data_dir_name

        logger.info("Project root resolvido em: %s", self.project_root)
        logger.info("Data root resolvido em: %s", self.data_root)

    def load_csv(self, filename: str, layer: str = "processed") -> pd.DataFrame:
        """
        Carrega um arquivo CSV da camada especificada.

        Args:
            filename (str): Nome do arquivo CSV
            layer (str): Camada de dados (raw, processed, external)

        Returns:
            pd.DataFrame: DataFrame carregado
        """
        try:
            file_path = self.data_root / layer / filename

            logger.info("Carregando dados de: %s", file_path)

            if not file_path.exists():
                raise FileNotFoundError(
                    f"Arquivo não encontrado: {file_path}"
                )

            df = pd.read_csv(file_path)

            logger.info(
                "Dados carregados com sucesso | shape=%s",
                df.shape
            )

            return df

        except Exception as error:
            logger.error(
                "Erro ao carregar dados",
                exc_info=True
            )
            raise MLProjectException(str(error)) from error

    @staticmethod
    def _find_project_root() -> Path:
        """
        Localiza a raiz do projeto (onde existe o diretório 'data').

        Returns:
            Path: Caminho da raiz do projeto
        """
        current = Path.cwd()

        while current != current.parent:
            if (current / "data").exists():
                return current
            current = current.parent

        raise MLProjectException(
            "Não foi possível localizar a raiz do projeto "
            "(diretório 'data' não encontrado)."
        )

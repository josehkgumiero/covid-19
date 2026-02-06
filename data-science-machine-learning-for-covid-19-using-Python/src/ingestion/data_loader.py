"""
data_loader.py

Módulo responsável por carregar datasets localizados na pasta /data
considerando as camadas: raw, clean, processed e storage.
"""

from pathlib import Path
from typing import Optional
import pandas as pd


class DataLoader:
    """
    Classe responsável por localizar e carregar arquivos de dados
    a partir das camadas do data lake local.
    """

    VALID_LAYERS = {"raw", "clean", "processed", "storage"}

    def __init__(self, project_root: Optional[Path] = None) -> None:
        """
        Inicializa o carregador de dados.

        Args:
            project_root (Optional[Path]): Caminho raiz do projeto.
                                           Se None, o caminho é resolvido automaticamente.
        """
        self.project_root: Path = (
            project_root if project_root else self._resolve_project_root()
        )
        self.data_dir: Path = self.project_root / "data"

    def _resolve_project_root(self) -> Path:
        """
        Resolve automaticamente a raiz do projeto assumindo
        que o script está em src/ e os notebooks em notebook/.

        Returns:
            Path: Caminho da raiz do projeto.
        """
        return Path(__file__).resolve().parents[2]

    def load_csv(
        self,
        filename: str,
        layer: str = "raw",
        encoding: str = "utf-8",
        sep: str = ","
    ) -> pd.DataFrame:
        """
        Carrega um arquivo CSV de uma camada específica do data lake.

        Args:
            filename (str): Nome do arquivo CSV.
            layer (str): Camada dos dados (raw, clean, processed, storage).
            encoding (str): Codificação do arquivo.
            sep (str): Separador do CSV.

        Returns:
            pd.DataFrame: DataFrame carregado.
        """
        try:
            if layer not in self.VALID_LAYERS:
                raise ValueError(
                    f"Invalid data layer '{layer}'. "
                    f"Valid layers are: {self.VALID_LAYERS}"
                )

            layer_dir: Path = self.data_dir / layer

            if not layer_dir.exists():
                raise FileNotFoundError(f"Data layer directory not found: {layer_dir}")

            file_path: Path = layer_dir / filename

            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            return pd.read_csv(file_path, encoding=encoding, sep=sep)

        except Exception as error:
            raise RuntimeError(
                f"Error loading CSV file from layer '{layer}': {error}"
            ) from error

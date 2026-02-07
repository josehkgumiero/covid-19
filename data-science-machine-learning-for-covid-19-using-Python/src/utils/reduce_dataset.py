"""
reduce_dataset.py

Reduz o tamanho de um dataset CSV localizado em data/raw/,
salva a versão reduzida em data/processed/ e copia o
dataset final para data/storage/.
"""

from pathlib import Path
import logging
import shutil
import pandas as pd


# =========================
# Logging configuration
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


class DatasetStorageManager:
    """
    Classe responsável por gerenciar o armazenamento final de datasets.
    """

    def __init__(self, storage_dir: Path) -> None:
        """
        Inicializa o gerenciador de storage.

        Args:
            storage_dir (Path): Diretório de armazenamento final.
        """
        self.storage_dir: Path = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def copy_to_storage(self, file_path: Path) -> Path:
        """
        Copia um arquivo para a pasta de storage.

        Args:
            file_path (Path): Caminho do arquivo a ser copiado.

        Returns:
            Path: Caminho do arquivo copiado no storage.
        """
        try:
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            destination: Path = self.storage_dir / file_path.name
            shutil.copy2(file_path, destination)

            logger.info(
                "Dataset copied to storage: %s",
                destination
            )

            return destination

        except Exception as error:
            logger.error(
                "Error copying dataset to storage: %s",
                error,
                exc_info=True
            )
            raise RuntimeError("Dataset storage copy failed.") from error


class DatasetReducer:
    """
    Classe responsável por reduzir o tamanho de datasets CSV
    respeitando um limite máximo de tamanho em MB.
    """

    def __init__(
        self,
        project_root: Path,
        input_filename: str,
        output_filename: str,
        max_size_mb: int = 90,
        random_state: int = 42
    ) -> None:
        self.project_root: Path = project_root
        self.raw_dir: Path = self.project_root / "data" / "raw"
        self.processed_dir: Path = self.project_root / "data" / "processed"
        self.storage_dir: Path = self.project_root / "data" / "storage"

        self.input_file: Path = self.raw_dir / input_filename
        self.output_file: Path = self.processed_dir / output_filename

        self.max_size_mb: int = max_size_mb
        self.random_state: int = random_state

        self.storage_manager = DatasetStorageManager(self.storage_dir)

    @staticmethod
    def _get_file_size_mb(file_path: Path) -> float:
        """Retorna o tamanho do arquivo em MB."""
        return file_path.stat().st_size / (1024 * 1024)

    def reduce(self) -> None:
        """Executa a redução do dataset e copia para storage."""
        try:
            if not self.input_file.exists():
                raise FileNotFoundError(
                    f"Input dataset not found: {self.input_file}"
                )

            self.processed_dir.mkdir(parents=True, exist_ok=True)

            original_size = self._get_file_size_mb(self.input_file)
            logger.info("Original dataset size: %.2f MB", original_size)

            if original_size <= self.max_size_mb:
                logger.info(
                    "Dataset already within size limit (%d MB). Skipping reduction.",
                    self.max_size_mb
                )
                final_file = self.input_file
            else:
                reduction_ratio = self.max_size_mb / original_size
                logger.info(
                    "Applying sampling ratio: %.4f to fit within %d MB",
                    reduction_ratio,
                    self.max_size_mb
                )

                df = pd.read_csv(self.input_file)

                reduced_df = df.sample(
                    frac=reduction_ratio,
                    random_state=self.random_state
                )

                reduced_df.to_csv(self.output_file, index=False)
                final_file = self.output_file

                final_size = self._get_file_size_mb(final_file)
                logger.info(
                    "Reduced dataset saved at: %s (%.2f MB)",
                    final_file,
                    final_size
                )

            # Copia dataset final para storage
            self.storage_manager.copy_to_storage(final_file)

        except Exception as error:
            logger.error(
                "Error during dataset reduction pipeline: %s",
                error,
                exc_info=True
            )
            raise RuntimeError("Dataset reduction pipeline failed.") from error


def main() -> None:
    """Função principal."""
    try:
        # src/transformation/reduce_dataset.py → src → project root
        project_root: Path = Path(__file__).resolve().parents[2]

        reducer = DatasetReducer(
            project_root=project_root,
            input_filename="corona_tested_individuals_ver_0083.english.csv",
            output_filename="corona_tested_individuals_reduced.csv",
            max_size_mb=90
        )

        reducer.reduce()

    except Exception as error:
        logger.critical(
            "Fatal error during dataset pipeline execution: %s",
            error,
            exc_info=True
        )


if __name__ == "__main__":
    main()

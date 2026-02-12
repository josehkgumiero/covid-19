import logging
import shutil
from pathlib import Path
from typing import List


class GitHubDatasetBuilder:
    """
    Cria uma versão reduzida e BALANCEADA do dataset
    para push no GitHub sem alterar o dataset original.
    """

    def __init__(
        self,
        raw_dir: str,
        github_dir: str,
        max_repo_size_mb: int = 90
    ):
        self.raw_dir = Path(raw_dir)
        self.github_dir = Path(github_dir)
        self.max_repo_size_bytes = max_repo_size_mb * 1024 * 1024
        self._configure_logger()

    def _configure_logger(self):
        self.logger = logging.getLogger("GitHubDatasetBuilder")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def get_class_folders(self) -> List[Path]:
        return [f for f in self.raw_dir.iterdir() if f.is_dir()]

    def copy_file(self, source: Path):
        relative_path = source.relative_to(self.raw_dir)
        destination = self.github_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    def build_dataset(self):
        self.logger.info("Criando dataset reduzido e balanceado...")

        # Limpa pasta se existir
        if self.github_dir.exists():
            shutil.rmtree(self.github_dir)

        self.github_dir.mkdir(parents=True, exist_ok=True)

        class_folders = self.get_class_folders()
        num_classes = len(class_folders)

        if num_classes == 0:
            self.logger.warning("Nenhuma classe encontrada.")
            return

        # Divide o tamanho igualmente entre as classes
        size_per_class = self.max_repo_size_bytes // num_classes

        for folder in class_folders:
            self.logger.info(f"Processando classe: {folder.name}")

            files = list(folder.glob("*"))
            files.sort(key=lambda f: f.stat().st_size)  # menores primeiro

            current_class_size = 0

            for file in files:
                file_size = file.stat().st_size

                if current_class_size + file_size > size_per_class:
                    break

                self.copy_file(file)
                current_class_size += file_size

            self.logger.info(
                f"Classe {folder.name} copiada com "
                f"{current_class_size / (1024*1024):.2f} MB"
            )

        self.logger.info("Dataset GitHub criado com sucesso.")

    def execute(self):
        try:
            self.build_dataset()
        except Exception as e:
            self.logger.error(f"Erro durante execução: {e}")
            raise


if __name__ == "__main__":
    builder = GitHubDatasetBuilder(
        raw_dir="data/raw",
        github_dir="data/github_dataset",
        max_repo_size_mb=90
    )
    builder.execute()

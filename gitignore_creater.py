"""
gitignore_creator.py

Gera automaticamente um .gitignore na raiz do projeto
(COVID-19) e cria as subpastas padrão dentro de cada
projeto caso não existam.
"""

from pathlib import Path
from typing import List


class GitignoreGenerator:
    """
    Responsável por gerar o .gitignore na raiz do projeto
    e garantir que as subpastas de dados existam.
    """

    def __init__(self, project_root: Path) -> None:
        self.project_root: Path = project_root
        self.gitignore_path: Path = self.project_root / ".gitignore"

    def _list_project_directories(self) -> List[Path]:
        """
        Lista diretórios na raiz ignorando ocultos e .git.
        """
        return [
            item for item in self.project_root.iterdir()
            if item.is_dir()
            and not item.name.startswith(".")
            and item.name != ".git"
        ]

    def _ensure_data_subfolders(self, project_dir: Path) -> None:
        """
        Garante que as subpastas data/raw, data/processed
        e data/storage existam.
        """
        data_root = project_dir / "data"
        subfolders = ["raw", "processed", "storage"]

        for folder in subfolders:
            (data_root / folder).mkdir(parents=True, exist_ok=True)

    def _build_ignore_patterns(self) -> List[str]:
        """
        Constrói as regras do .gitignore.
        """
        base_patterns: List[str] = [
            "# =========================",
            "# Python",
            "# =========================",
            "__pycache__/",
            "*.py[cod]",
            "*.pyd",
            "*.pyo",
            "",
            "# =========================",
            "# Virtual Environments",
            "# =========================",
            "env/",
            "venv/",
            ".venv/",
            ".env/",
            "",
            "# =========================",
            "# Jupyter",
            "# =========================",
            ".ipynb_checkpoints/",
            "",
            "# =========================",
            "# IDEs",
            "# =========================",
            ".vscode/",
            ".idea/",
            "",
            "# =========================",
            "# Logs",
            "# =========================",
            "*.log",
            "logs/",
            "",
            "# =========================",
            "# Project Data Rules",
            "# =========================",
        ]

        data_patterns: List[str] = []

        for project_dir in self._list_project_directories():

            # Garante criação das subpastas
            self._ensure_data_subfolders(project_dir)

            data_patterns.extend([
                f"{project_dir.name}/data/raw/",
                f"{project_dir.name}/data/storage/",
                f"{project_dir.name}/data/*",
                f"!{project_dir.name}/data/processed/",
                f"!{project_dir.name}/data/github_dataset/",
                "",
            ])

        return base_patterns + data_patterns

    def create_gitignore(self, overwrite: bool = False) -> None:
        """
        Cria ou sobrescreve o .gitignore.
        """
        if self.gitignore_path.exists() and not overwrite:
            print(".gitignore já existe. Use overwrite=True para substituir.")
            return

        patterns = self._build_ignore_patterns()

        with self.gitignore_path.open("w", encoding="utf-8") as file:
            file.write("\n".join(patterns))

        print(
            f".gitignore criado em {self.gitignore_path} "
            f"com {len(patterns)} regras."
        )


def main() -> None:
    """
    Executa o gerador assumindo que o script
    está na raiz do projeto.
    """
    project_root = Path.cwd()
    generator = GitignoreGenerator(project_root)
    generator.create_gitignore(overwrite=True)


if __name__ == "__main__":
    main()

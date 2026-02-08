"""
create_gitignore.py

Cria um arquivo .gitignore na raiz do repositório Git (COVID-19),
listando automaticamente os diretórios abaixo da raiz e adicionando
regras no formato: <diretorio>/data/
"""

from pathlib import Path
from typing import List


class GitRepositoryNotFoundError(Exception):
    """Erro levantado quando a raiz do repositório Git não é encontrada."""
    pass


class GitignoreGenerator:
    """
    Classe responsável por localizar a raiz do repositório Git,
    identificar diretórios de projetos e gerar o arquivo .gitignore.
    """

    def __init__(self, start_path: Path) -> None:
        """
        Inicializa o gerador.

        Args:
            start_path (Path): Caminho inicial para busca da raiz Git.
        """
        self.start_path: Path = start_path
        self.repo_root: Path = self._find_git_root()
        self.gitignore_path: Path = self.repo_root / ".gitignore"

    def _find_git_root(self) -> Path:
        """
        Procura a raiz do repositório Git (diretório que contém .git).

        Returns:
            Path: Caminho da raiz do repositório.

        Raises:
            GitRepositoryNotFoundError: Se a raiz não for encontrada.
        """
        current: Path = self.start_path

        while current != current.parent:
            if (current / ".git").is_dir():
                return current
            current = current.parent

        raise GitRepositoryNotFoundError(
            "Git repository root (.git directory) not found."
        )

    def _list_project_directories(self) -> List[Path]:
        """
        Lista os diretórios imediatamente abaixo da raiz do repositório,
        ignorando pastas ocultas e a pasta .git.

        Returns:
            List[Path]: Lista de diretórios de projeto.
        """
        return [
            item for item in self.repo_root.iterdir()
            if item.is_dir() and not item.name.startswith(".") and item.name != ".git"
        ]

    def _build_ignore_patterns(self) -> List[str]:
        """
        Constrói as regras do .gitignore, incluindo regras avançadas
        para diretórios de dados (data), ignorando tudo e liberando
        apenas subpastas específicas.
        """
        base_patterns: List[str] = [
            "# =========================",
            "# Python",
            "# =========================",
            "__pycache__/",
            "*.py[cod]",
            "*.pyd",
            "*.pyo",
            ".Python",
            "",
            "# =========================",
            "# Virtual environments",
            "# =========================",
            "env/",
            "venv/",
            ".venv/",
            ".env/",
            "",
            "# =========================",
            "# Jupyter Notebook",
            "# =========================",
            ".ipynb_checkpoints/",
            "",
            "# =========================",
            "# IDEs and editors",
            "# =========================",
            ".vscode/",
            ".idea/",
            "*.code-workspace",
            "",
            "# =========================",
            "# Operating system files",
            "# =========================",
            ".DS_Store",
            "Thumbs.db",
            "desktop.ini",
            "",
            "# =========================",
            "# Logs",
            "# =========================",
            "*.log",
            "logs/",
            "",
            "# =========================",
            "# Build and distribution",
            "# =========================",
            "build/",
            "dist/",
            "*.egg-info/",
            "",
            "# =========================",
            "# Cache and temporary files",
            "# =========================",
            ".cache/",
            "tmp/",
            "temp/",
            "",
            "# =========================",
            "# Testing and coverage",
            "# =========================",
            ".pytest_cache/",
            ".coverage",
            "htmlcov/",
            "",
            "# =========================",
            "# Project data directories",
            "# =========================",
        ]

        project_dirs: List[Path] = self._list_project_directories()

        data_patterns: List[str] = []

        for project_dir in project_dirs:
            data_patterns.extend([
                f"{project_dir.name}/data/*",
                f"!{project_dir.name}/data/processed/",
                f"!{project_dir.name}/data/storage/",
                f"{project_dir.name}/data/raw/",
                "",
            ])

        return base_patterns + data_patterns


    def create_gitignore(self) -> bool:
        """
        Cria o arquivo .gitignore na raiz do repositório.

        Returns:
            bool: True se criado com sucesso, False caso contrário.
        """
        try:
            if self.gitignore_path.exists():
                raise FileExistsError(
                    f".gitignore already exists at {self.gitignore_path}"
                )

            patterns: List[str] = self._build_ignore_patterns()

            with self.gitignore_path.open(mode="w", encoding="utf-8") as file:
                file.write("\n".join(patterns))

            print(
                f".gitignore created at {self.gitignore_path} "
                f"with {len(patterns)} rules."
            )
            return True

        except FileExistsError as error:
            print(f"Warning: {error}")
            return False

        except PermissionError:
            print("Error: Permission denied while creating .gitignore.")
            return False

        except Exception as error:
            print(f"Unexpected error: {error}")
            return False


def main() -> None:
    """
    Função principal do script.
    """
    try:
        script_dir: Path = Path(__file__).resolve().parent
        generator: GitignoreGenerator = GitignoreGenerator(script_dir)
        generator.create_gitignore()

    except GitRepositoryNotFoundError as error:
        print(f"Fatal error: {error}")

    except Exception as error:
        print(f"Unexpected fatal error: {error}")


if __name__ == "__main__":
    main()

"""
python_environment_checker.py

Script responsável por identificar qual interpretador Python
está sendo utilizado no ambiente atual (terminal ou Jupyter Notebook).
"""

import sys
from typing import Optional


class PythonEnvironmentChecker:
    """
    Classe responsável por verificar informações do ambiente Python.
    """

    def __init__(self) -> None:
        """
        Inicializa o verificador de ambiente.
        """
        pass

    def get_python_executable(self) -> Optional[str]:
        """
        Obtém o caminho completo do executável Python em uso.

        Returns:
            Optional[str]: Caminho do executável Python ou None em caso de erro.
        """
        try:
            executable_path: str = sys.executable

            if not executable_path:
                raise ValueError("Python executable path is empty.")

            return executable_path

        except Exception as error:
            print(f"Error while retrieving Python executable: {error}")
            return None


def print_python_executable() -> None:
    """
    Função responsável por imprimir o executável Python ativo.
    """
    try:
        checker: PythonEnvironmentChecker = PythonEnvironmentChecker()
        executable: Optional[str] = checker.get_python_executable()

        if executable:
            print(f"Active Python executable:\n{executable}")
        else:
            print("Unable to determine the active Python executable.")

    except Exception as error:
        print(f"Unexpected error: {error}")


if __name__ == "__main__":
    print_python_executable()

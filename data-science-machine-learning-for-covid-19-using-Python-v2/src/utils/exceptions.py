"""
exceptions.py

Define exceções customizadas para o projeto de Machine Learning.
Centraliza erros de domínio para facilitar rastreamento, logging
e tratamento consistente.
"""


class MLProjectException(Exception):
    """
    Exceção base para erros do projeto de Machine Learning.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)

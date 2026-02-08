"""
settings.py

Configurações globais do projeto de Machine Learning.
Centraliza hiperparâmetros, seeds e opções de execução.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """
    Configurações do modelo de ML.
    """
    # Split
    test_size: float = 0.2
    random_state: int = 42

    # Gradient Boosting
    learning_rate: float = 0.2
    n_estimators: int = 200
    max_depth: int = 3

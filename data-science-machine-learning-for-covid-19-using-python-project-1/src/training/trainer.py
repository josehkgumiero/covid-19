"""
trainer.py

Responsável pelo treinamento do modelo de Machine Learning.
Inclui split estratificado, validações, logging e tratamento de exceções.
"""

from typing import Tuple
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

from utils.logger import get_logger
from utils.exceptions import MLProjectException
from config.settings import ModelConfig


logger = get_logger(__name__)


class ModelTrainer:
    """
    Classe responsável pela etapa de treinamento do modelo.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.model: GradientBoostingClassifier | None = None

        # Estado exposto para avaliação
        self.X_train: pd.DataFrame | None = None
        self.X_test: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None
        self.y_test: pd.Series | None = None

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Executa o treinamento do modelo.

        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target

        Returns:
            Tuple contendo X_train, X_test, y_train, y_test
        """
        try:
            self._validate_target(y)

            logger.info("Iniciando split de treino e teste (estratificado)")

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X,
                y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=y
            )

            self._log_class_distribution(self.y_train, self.y_test)

            logger.info("Inicializando GradientBoostingClassifier")

            self.model = GradientBoostingClassifier(
                learning_rate=self.config.learning_rate,
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth
            )

            logger.info("Treinando modelo")
            self.model.fit(self.X_train, self.y_train)

            logger.info("Treinamento concluído com sucesso")

            return self.X_train, self.X_test, self.y_train, self.y_test

        except Exception as error:
            logger.error(
                "Erro durante o treinamento do modelo",
                exc_info=True
            )
            raise MLProjectException(str(error)) from error

    # =========================
    # Métodos internos
    # =========================
    def _validate_target(self, y: pd.Series) -> None:
        """
        Valida se o target possui pelo menos duas classes.
        """
        unique_classes = y.unique()

        if len(unique_classes) < 2:
            raise MLProjectException(
                "Target possui menos de 2 classes. "
                "Classificação requer pelo menos duas."
            )

        logger.info(
            "Target validado | Classes encontradas: %s",
            unique_classes
        )

    def _log_class_distribution(
        self,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> None:
        """
        Loga a distribuição de classes nos conjuntos de treino e teste.
        """
        logger.info(
            "Distribuição de classes no treino: %s",
            Counter(y_train)
        )
        logger.info(
            "Distribuição de classes no teste: %s",
            Counter(y_test)
        )

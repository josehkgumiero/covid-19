"""
trainer.py

Módulo responsável pelo treinamento do modelo de Machine Learning.

Responsabilidades:
- Validação do target
- Split estratificado
- Logging detalhado
- Tratamento de exceções
- Armazenamento de estado (X_train, X_test, y_train, y_test)
- Inicialização e treinamento do modelo
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
    Encapsula todo o estado necessário para avaliação posterior.
    """

    def __init__(self, config: ModelConfig) -> None:
        """
        Inicializa o treinador com as configurações do modelo.

        Args:
            config (ModelConfig): Configurações do modelo
        """
        self.config: ModelConfig = config
        self.model: GradientBoostingClassifier | None = None

        # Estado do treino (exposto para avaliação)
        self.X_train: pd.DataFrame | None = None
        self.X_test: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None
        self.y_test: pd.Series | None = None

    # =========================
    # API pública
    # =========================
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Executa o fluxo completo de treinamento do modelo.

        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
                X_train, X_test, y_train, y_test
        """
        try:
            logger.info("Iniciando processo de treinamento")

            self._validate_inputs(X, y)
            self._validate_target(y)

            self._split_data(X, y)
            self._log_class_distribution()

            self._initialize_model()
            self._fit_model()

            logger.info("Treinamento concluído com sucesso")

            return self.X_train, self.X_test, self.y_train, self.y_test

        except Exception as error:
            logger.error(
                "Erro durante o treinamento do modelo",
                exc_info=True
            )
            raise MLProjectException(error) from error

    # =========================
    # Métodos internos
    # =========================
    def _validate_inputs(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Valida os tipos de entrada.
        """
        if not isinstance(X, pd.DataFrame):
            raise MLProjectException("X must be a pandas DataFrame.")

        if not isinstance(y, pd.Series):
            raise MLProjectException("y must be a pandas Series.")

        if X.empty or y.empty:
            raise MLProjectException("X and y must not be empty.")

        logger.info("Entradas validadas com sucesso")

    def _validate_target(self, y: pd.Series) -> None:
        """
        Valida se o target possui pelo menos duas classes.
        """
        unique_classes = y.dropna().unique()

        if len(unique_classes) < 2:
            raise MLProjectException(
                "Target variable has less than 2 classes. "
                "Classification requires at least two classes."
            )

        logger.info(
            "Target validado | Classes encontradas: %s",
            unique_classes
        )

    def _split_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Realiza o split estratificado e armazena o estado internamente.
        """
        logger.info("Realizando split estratificado de treino e teste")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )

    def _log_class_distribution(self) -> None:
        """
        Loga a distribuição de classes nos conjuntos de treino e teste.
        """
        logger.info(
            "Distribuição de classes no treino: %s",
            Counter(self.y_train)
        )
        logger.info(
            "Distribuição de classes no teste: %s",
            Counter(self.y_test)
        )

    def _initialize_model(self) -> None:
        """
        Inicializa o modelo de Machine Learning.
        """
        logger.info("Inicializando GradientBoostingClassifier")

        self.model = GradientBoostingClassifier(
            learning_rate=self.config.learning_rate,
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth
        )

    def _fit_model(self) -> None:
        """
        Executa o treinamento do modelo.
        """
        if self.model is None:
            raise MLProjectException("Model has not been initialized.")

        logger.info("Treinando modelo")
        self.model.fit(self.X_train, self.y_train)

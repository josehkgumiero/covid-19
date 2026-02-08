"""
ml_pipeline.py

Pipeline completo de Machine Learning.

Responsabilidades:
- Seleção de features
- Separação de X e y
- Treinamento do modelo
- Orquestração do fluxo de ML

Suporta dois modos:
1. run(df)              -> retorna previsões
2. run_with_trainer(df) -> retorna o ModelTrainer treinado
"""

import pandas as pd

from preprocessing.feature_selector import FeatureSelector
from training.trainer import ModelTrainer
from inference.predictor import Predictor
from config.settings import ModelConfig


class MLPipeline:
    """
    Pipeline completo de Machine Learning.
    """

    def __init__(self) -> None:
        self.config = ModelConfig()

    def run(self, df: pd.DataFrame):
        """
        Executa o pipeline completo e retorna apenas as previsões.

        Args:
            df (pd.DataFrame): Dataset completo

        Returns:
            array-like: Previsões do modelo
        """
        features = [
            "cough",
            "fever",
            "sore_throat",
            "shortness_of_breath",
            "head_ache",
            "age_60_and_above",
            "gender",
            "contact_with_confirmed",
        ]

        selector = FeatureSelector(features, "corona_result")
        X, y = selector.transform(df)

        trainer = ModelTrainer(self.config)
        X_train, X_test, y_train, y_test = trainer.train(X, y)

        predictor = Predictor(trainer.model)
        predictions = predictor.predict(X_test)

        return predictions

    def run_with_trainer(self, df: pd.DataFrame) -> ModelTrainer:
        """
        Executa o pipeline completo e retorna o trainer treinado.

        Args:
            df (pd.DataFrame): Dataset completo

        Returns:
            ModelTrainer: Trainer treinado
        """
        features = [
            "cough",
            "fever",
            "sore_throat",
            "shortness_of_breath",
            "head_ache",
            "age_60_and_above",
            "gender",
            "contact_with_confirmed",
        ]

        selector = FeatureSelector(features, "corona_result")
        X, y = selector.transform(df)

        trainer = ModelTrainer(self.config)
        trainer.train(X, y)

        return trainer

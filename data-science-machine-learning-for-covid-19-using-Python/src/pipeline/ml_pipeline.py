import pandas as pd
from preprocessing.feature_selector import FeatureSelector
from training.trainer import ModelTrainer
from inference.predictor import Predictor
from config.settings import ModelConfig

class MLPipeline:
    """Pipeline completo de Machine Learning."""

    def __init__(self):
        self.config = ModelConfig()
        """
        ml_pipeline.py
        
        Pipeline completo de Machine Learning.
        
        Responsabilidades:
        - Seleção de features
        - Separação de X e y
        - Treinamento do modelo
        - Predição
        - Orquestração do fluxo de ML
        
        Este pipeline suporta DOIS modos de execução:
        1. run(df)              -> retorna previsões (modo simples)
        2. run_with_trainer(df) -> retorna o ModelTrainer treinado (modo avançado)
        """

from typing import Tuple

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

    # ==========================================================
    # MODO 1 — comportamento ORIGINAL (NÃO ALTERADO)
    # ==========================================================
    def run(self, df: pd.DataFrame):
        """
        Executa o pipeline completo e retorna apenas as previsões.
        (Mantém o comportamento original do código.)

        Args:
            df (pd.DataFrame): Dataset completo

        Returns:
            pd.Series | np.ndarray: Previsões do modelo
        """
        features = [
            'cough',
            'fever',
            'sore_throat',
            'shortness_of_breath',
            'head_ache',
            'age_60_and_above',
            'gender',
            'contact_with_confirmed'
        ]

        selector = FeatureSelector(features, 'corona_result')
        X, y = selector.transform(df)

        trainer = ModelTrainer(self.config)
        X_train, X_test, y_train, y_test = trainer.train(X, y)

        predictor = Predictor(trainer.model)
        predictions = predictor.predict(X_test)

        return predictions

    # ==========================================================
    # MODO 2 — novo comportamento (AVANÇADO / PROFISSIONAL)
    # ==========================================================
    def run_with_trainer(self, df: pd.DataFrame) -> ModelTrainer:
        """
        Executa o pipeline completo e retorna o ModelTrainer treinado.
        Permite acesso a:
        - trainer.model
        - trainer.X_test
        - trainer.y_test
        - avaliação e persistência

        Args:
            df (pd.DataFrame): Dataset completo

        Returns:
            ModelTrainer: Trainer treinado
        """
        features = [
            'cough',
            'fever',
            'sore_throat',
            'shortness_of_breath',
            'head_ache',
            'age_60_and_above',
            'gender',
            'contact_with_confirmed'
        ]

        selector = FeatureSelector(features, 'corona_result')
        X, y = selector.transform(df)

        trainer = ModelTrainer(self.config)
        trainer.train(X, y)

        return trainer


    def run(self, df: pd.DataFrame):
        features = [
            'cough', 'fever', 'sore_throat', 'shortness_of_breath',
            'head_ache', 'age_60_and_above', 'gender',
            'contact_with_confirmed'
        ]

        selector = FeatureSelector(features, 'corona_result')
        X, y = selector.transform(df)

        trainer = ModelTrainer(self.config)
        X_train, X_test, y_train, y_test = trainer.train(X, y)

        predictor = Predictor(trainer.model)
        predictions = predictor.predict(X_test)

        return predictions

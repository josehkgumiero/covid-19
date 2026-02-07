"""
evaluator.py

Módulo responsável pela avaliação de modelos de Machine Learning.
Inclui:
- Métricas de classificação
- Matriz de confusão
- Relatório de classificação
- Curva ROC
"""

from typing import Dict
import logging

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

from utils.logger import get_logger
from utils.exceptions import MLProjectException


logger = get_logger(__name__)


class ModelEvaluator:
    """
    Classe responsável pela avaliação de modelos de classificação.
    """

    def evaluate_metrics(
        self,
        y_true: pd.Series,
        y_pred: pd.Series
    ) -> Dict[str, float]:
        """
        Calcula métricas básicas de classificação.

        Args:
            y_true (pd.Series): Valores reais
            y_pred (pd.Series): Valores previstos

        Returns:
            Dict[str, float]: Métricas calculadas
        """
        try:
            logger.info("Calculando métricas de avaliação")

            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "recall": recall_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred)
            }

            for name, value in metrics.items():
                logger.info("%s: %.2f%%", name.capitalize(), value * 100)

            return metrics

        except Exception as error:
            logger.error("Erro ao calcular métricas", exc_info=True)
            raise MLProjectException(error) from error

    def classification_report(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        target_names: list[str] | None = None
    ) -> str:
        """
        Gera relatório de classificação.

        Args:
            y_true (pd.Series): Valores reais
            y_pred (pd.Series): Valores previstos
            target_names (list[str], optional): Nomes das classes

        Returns:
            str: Relatório de classificação
        """
        try:
            logger.info("Gerando classification report")

            return classification_report(
                y_true,
                y_pred,
                target_names=target_names
            )

        except Exception as error:
            logger.error("Erro ao gerar classification report", exc_info=True)
            raise MLProjectException(error) from error

    def plot_confusion_matrix(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        labels: list[str] | None = None
    ) -> None:
        """
        Plota a matriz de confusão.

        Args:
            y_true (pd.Series): Valores reais
            y_pred (pd.Series): Valores previstos
            labels (list[str], optional): Labels das classes
        """
        try:
            logger.info("Plotando matriz de confusão")

            conf_mat = confusion_matrix(y_true, y_pred)

            fig, ax = plt.subplots()
            cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
            fig.colorbar(cax)

            if labels:
                ax.set_xticklabels([""] + labels)
                ax.set_yticklabels([""] + labels)

            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

            plt.show()

        except Exception as error:
            logger.error("Erro ao plotar matriz de confusão", exc_info=True)
            raise MLProjectException(error) from error

    def plot_roc_curve(
        self,
        model,
        X_test: pd.DataFrame,
        y_true: pd.Series
    ) -> None:
        """
        Plota a curva ROC.

        Args:
            model: Modelo treinado
            X_test (pd.DataFrame): Features de teste
            y_true (pd.Series): Valores reais
        """
        try:
            logger.info("Plotando curva ROC")

            if not hasattr(model, "predict_proba"):
                raise MLProjectException(
                    "Model does not support probability prediction."
                )

            y_proba = model.predict_proba(X_test)[:, 1]

            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)

            plt.figure()
            plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend(loc="lower right")
            plt.show()

        except Exception as error:
            logger.error("Erro ao plotar curva ROC", exc_info=True)
            raise MLProjectException(error) from error

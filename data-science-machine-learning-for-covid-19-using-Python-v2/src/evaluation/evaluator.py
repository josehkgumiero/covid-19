"""
evaluator.py

Responsável pela avaliação de modelos de Machine Learning.
Inclui métricas, relatórios, matriz de confusão e curva ROC.
"""

from typing import Dict, List
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
)

from utils.logger import get_logger
from utils.exceptions import MLProjectException


logger = get_logger(__name__)


class ModelEvaluator:
    """
    Avaliador de modelos de classificação.
    """

    def evaluate_metrics(
        self,
        y_true,
        y_pred
    ) -> Dict[str, float]:
        """
        Calcula métricas básicas de classificação.

        Returns:
            Dict[str, float]: Métricas calculadas
        """
        try:
            logger.info("Calculando métricas de avaliação")

            return {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1_score": f1_score(y_true, y_pred, zero_division=0),
            }

        except Exception as error:
            logger.error(
                "Erro ao calcular métricas",
                exc_info=True
            )
            raise MLProjectException(str(error)) from error

    def classification_report(
        self,
        y_true,
        y_pred,
        target_names: List[str]
    ) -> str:
        """
        Gera o classification report.
        """
        try:
            logger.info("Gerando classification report")

            return classification_report(
                y_true,
                y_pred,
                target_names=target_names
            )

        except Exception as error:
            logger.error(
                "Erro ao gerar classification report",
                exc_info=True
            )
            raise MLProjectException(str(error)) from error

    def plot_confusion_matrix(
        self,
        y_true,
        y_pred,
        labels: List[str]
    ) -> None:
        """
        Plota a matriz de confusão.
        """
        try:
            logger.info("Plotando matriz de confusão")

            cm = confusion_matrix(y_true, y_pred)

            plt.figure(figsize=(6, 5))
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            plt.xticks(range(len(labels)), labels)
            plt.yticks(range(len(labels)), labels)

            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix")

            for i in range(len(labels)):
                for j in range(len(labels)):
                    plt.text(j, i, cm[i, j], ha="center", va="center")

            plt.tight_layout()
            plt.show()

        except Exception as error:
            logger.error(
                "Erro ao plotar matriz de confusão",
                exc_info=True
            )
            raise MLProjectException(str(error)) from error

    def plot_roc_curve(
        self,
        model,
        X_test,
        y_true
    ) -> None:
        """
        Plota a curva ROC.
        """
        try:
            logger.info("Plotando curva ROC")

            RocCurveDisplay.from_estimator(
                model,
                X_test,
                y_true
            )

            plt.title("ROC Curve")
            plt.show()

        except Exception as error:
            logger.error(
                "Erro ao plotar curva ROC",
                exc_info=True
            )
            raise MLProjectException(str(error)) from error

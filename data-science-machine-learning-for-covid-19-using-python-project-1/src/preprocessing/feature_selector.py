"""
feature_selector.py

Responsável por selecionar features e target a partir de um DataFrame.
Inclui validações fortes para evitar erros de schema.
"""

from typing import List, Tuple
import pandas as pd

from utils.logger import get_logger
from utils.exceptions import MLProjectException


logger = get_logger(__name__)


class FeatureSelector:
    """
    Seleciona colunas de features e a variável alvo (target).
    """

    def __init__(
        self,
        features: List[str],
        target: str,
        strict: bool = False
    ) -> None:
        """
        Args:
            features (List[str]): Lista de nomes das features esperadas
            target (str): Nome da coluna alvo
            strict (bool):
                - True  -> falha se alguma feature estiver ausente
                - False -> ignora features ausentes (default)
        """
        self.features = features
        self.target = target
        self.strict = strict

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Seleciona X e y a partir do DataFrame.
        """
        try:
            logger.info("Selecionando features e target")

            # Validação: features devem ser strings
            invalid_features = [
                f for f in self.features if not isinstance(f, str)
            ]
            if invalid_features:
                raise MLProjectException(
                    f"Features inválidas (não são strings): {invalid_features}. "
                    "Verifique se valores de registros estão sendo usados como nomes de coluna."
                )

            # Validação do target
            if self.target not in df.columns:
                raise MLProjectException(
                    f"Target ausente no DataFrame: {self.target}"
                )

            available_features = [f for f in self.features if f in df.columns]
            missing_features = [f for f in self.features if f not in df.columns]

            if missing_features:
                message = f"Features ausentes ignoradas: {missing_features}"

                if self.strict:
                    raise MLProjectException(message)

                logger.warning(message)

            if not available_features:
                raise MLProjectException(
                    "Nenhuma feature válida encontrada no DataFrame."
                )

            X = df[available_features].copy()
            y = df[self.target].copy()

            logger.info(
                "Seleção concluída | X shape=%s | y shape=%s",
                X.shape,
                y.shape
            )

            logger.info(
                "Features utilizadas: %s",
                available_features
            )

            return X, y

        except Exception as error:
            logger.error(
                "Erro ao selecionar features/target",
                exc_info=True
            )
            raise MLProjectException(str(error)) from error

import pandas as pd
from typing import Tuple

class FeatureSelector:
    """Responsável por separar features e target."""

    def __init__(self, features: list, target: str):
        self.features = features
        self.target = target

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        X = df[self.features]
        y = df[self.target]
        return X, y

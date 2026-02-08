import logging
import numpy as np
import pandas as pd

from utils.logger import get_logger  # assumindo padrão do projeto


class FeatureEncoder:
    """
    Encodes categorical features into numerical representations
    using vectorized NumPy operations.
    """

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or get_logger(self.__class__.__name__)

    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature encodings to the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Encoded DataFrame.
        """
        try:
            self.logger.info("Starting feature encoding process")

            required_columns = {
                "gender",
                "age_60_and_above",
                "corona_result",
                "test_indication",
            }

            missing = required_columns - set(df.columns)
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            df = df.copy()

            self._encode_gender(df)
            self._encode_age(df)
            self._encode_corona_result(df)
            self._encode_contact_with_confirmed(df)

            self.logger.info("Feature encoding completed successfully")
            return df

        except Exception as e:
            self.logger.exception("Feature encoding failed")
            raise RuntimeError("Error during feature encoding") from e

    # =========================
    # Private encoding methods
    # =========================

    def _encode_gender(self, df: pd.DataFrame) -> None:
        self.logger.info("Encoding gender column")
        df["gender"] = np.where(
            df["gender"].str.lower() == "male", 1, 0
        )

    def _encode_age(self, df: pd.DataFrame) -> None:
        self.logger.info("Encoding age_60_and_above column")
        df["age_60_and_above"] = np.where(
            df["age_60_and_above"] == "Yes", 1, 0
        )

    def _encode_corona_result(self, df: pd.DataFrame) -> None:
        self.logger.info("Encoding corona_result column")
        df["corona_result"] = np.where(
            df["corona_result"].str.lower() == "positive", 1, 0
        )

    def _encode_contact_with_confirmed(self, df: pd.DataFrame) -> None:
        self.logger.info("Encoding contact_with_confirmed column")
        df["contact_with_confirmed"] = np.where(
            df["test_indication"] == "Contact with confirmed", 1, 0
        )

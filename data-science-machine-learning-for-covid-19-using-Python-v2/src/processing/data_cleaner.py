import logging
import pandas as pd


class DataCleaner:
    """
    Handles data cleaning operations following
    best practices for reproducibility and logging.
    """

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def drop_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes rows with missing values and logs shape changes.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        try:
            if df.empty:
                self.logger.warning("Received an empty DataFrame. No rows to drop.")
                return df

            initial_shape = df.shape
            self.logger.info(f"Initial DataFrame shape: {initial_shape}")

            cleaned_df = df.dropna()

            final_shape = cleaned_df.shape
            self.logger.info(f"Final DataFrame shape after dropna: {final_shape}")
            self.logger.info(
                f"Rows removed: {initial_shape[0] - final_shape[0]}"
            )

            return cleaned_df

        except Exception as e:
            self.logger.exception("Error while dropping missing values.")
            raise RuntimeError("Data cleaning failed.") from e

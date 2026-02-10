import pandas as pd
from src.utils.logger import get_logger
from src.utils.exceptions import DataLoadError

class DataLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.logger = get_logger(self.__class__.__name__)

    def load(self) -> pd.DataFrame:
        try:
            self.logger.info(f"Loading data from {self.filepath}")
            df = pd.read_csv(self.filepath)
            return df
        except Exception as e:
            self.logger.error("Failed to load dataset", exc_info=True)
            raise DataLoadError(str(e))

from dataclasses import dataclass

@dataclass
class ModelConfig:
    learning_rate: float = 0.2
    n_estimators: int = 200
    max_depth: int = 3
    test_size: float = 0.2
    random_state: int = 1
    
# Data Science & Machine Learning for COVID-19 using Python

## Course Introduction

- COVID-19 has been around for a while.
- During this time, various methodologies have been developed to tackle various problems related to COVID-19.
- This course aims to teach to various Data Science & Machine Learning concepts by solving various problems related to COVID-19.
- It's a project-based learning, which will help you understand how you can apply data science on real life datasets and not any dummy dataset.
- We will go through lot of concepts, frameworks, tools.

## Learning Objectives

- Work on complete end-to-end real-time projects related to COVID-19.
- Apply Deep Learning and Machine Learning concepts on real world use cases.
- Apply concepts like Time series Analysis, Dashboard Building, LSTMs, Object Detection, GBMs, Chatbot building.
- Work with various types of data which includes structured and unstructured data like text, image, videos.

## Course Structure

- Project based - each section is indepependent of each other
- Each project is divided into:
    - problem statement
    - approach
    - solution
        - covers different concept, framework, tools

### Classifying COVID-19 patients using symptoms

Problem - chest x-ray classification

Problem statement

-  To classify a given chest x-ray into one of the following classes:
1. COVID-19 (219)
2. Pneumonia (1345)
3. Normal (1341)

- Total Data Available for COVID-19 x-rays: 219

- Class imbalance
    - Why is this a problem? 
        - The model learns biased patterns 
            - A classifier may learn a biased rule such as: “If I mostly predict Pneumonia or Normal, I will achieve high accuracy.” In extreme cases, the model may never predict COVID-19 and still obtain a high overall accuracy.
        - Accuracy becomes a misleading metric 
            - The model may correctly classify most Normal and Pneumonia cases while misclassifying nearly all COVID-19 cases. As a result, the overall accuracy remains high, but the model is clinically unreliable.
        - Errors occur in the most critical class 
            - In this problem, COVID-19 is the most important class. False negatives (failing to detect COVID-19) are particularly severe and can have serious real-world consequences.
    - Inbalanced classification refers to a classification predictive modeling problem where the number of examples in the training dataset for each class label is not balanced.
        - That is, where the class distribution is not equal or close to equal, and is instead biased os skewed.
        - This imblanace cn be slight or strong. Depending on the sample size, ratios from 1:2 to 1:10 can be understood as a slight imblalance and ratios greater than 1:10 can be understood as a strong imbalance.
        - In both cases, the data with the class imbalance problem must be treated with specil techiques
        - Our data is slightly imbalanced with two classes having the same data. And one class having less data.

- The metric trap
    - One of major issues that novice users fall into when dealing with umbalanced datasets relates to the metrics used to evaluate their model. Using simples metrics like accuracy_score can be misleading.
    - In a dataset with higly unbalanced classes, if the claassifier always "predicts" the most common class without performing any analysis of the features, it will still have high accuracy rate, obviously ilusory.
    - Coming to our dataset, metric trap shouldn't be a big problem since we have two classes with almost equal examples so its not viable for the classifier to perform classification without any analysis on the feautres and just classify one class. It wouldn't lead to higher accuracy.
- Strategies for Handling Class Imbalance in Classification
    1. Understand class imbalance before modeling
        - Before any modeling step:
            - Analyze the class distribution
            - Identify the majority and minority classes
            - Assess the criticality of errors (e.g., false negatives)

    2. Use appropriate evaluation metrics (not only accuracy)

        - Avoid relying solely on:
        - Accuracy
        - Prefer:
            - Recall (especially for the minority class)
            - Precision
            - F1-score (macro / weighted)
            - Confusion Matrix
            - ROC-AUC per class
            - Precision-Recall AUC (PR-AUC)

    3. Data resampling techniques

        - Oversampling (minority class):
            - Random Oversampling
            - SMOTE
            - ADASYN
            - Undersampling (majority class):
            - Random Undersampling
            - Tomek Links
            - NearMiss
            - Note: Undersampling may lead to the loss of important information.

    4. Use class weights

        - This approach is highly effective when the dataset cannot be modified.
        - It penalizes misclassification of minority classes more strongly during training.

    5. Data augmentation (images)
        - Especially useful in computer vision tasks:
            - Rotation
            - Flipping
            - Zooming
            - Brightness and contrast adjustment
            - This increases the diversity of the minority class without collecting new data.

    6. Decision threshold tuning
        - Instead of using the default threshold (0.5):
            - Lower the threshold for the minority class
            - Increase recall
            - Reduce false negatives

    7. Use appropriate loss functions
        - Recommended for deep learning:
            - Focal Loss
            - Weighted Cross-Entropy
            - Balanced Loss
            - These loss functions force the model to focus on harder-to-classify classes.

    8. Proper validation strategy
        - Use Stratified K-Fold cross-validation
        - Preserve class proportions in both training and validation sets

    9. Ensemble methods focused on the minority class
        - Balanced bagging
        - Boosting with class weighting
        - Class-specific or specialized models


# Coding

## Update the PIP
```
python.exe -m pip install --upgrade pip
```

## Create gitignore file
```
python .\src\utils\gitignore_creater.py
```

## Reduce Dataset
```
src/utils/reduce_dataset.py
```

## Create environment venv
```
python -m venv .venv
```

## Active environment venv
```
.venv\Scripts\Activate.ps1
```

## Install dependencies
```
pip install -r requirements.txt
```

## Validate dependencies
```
python -c "import numpy, pandas, matplotlib, sklearn, imblearn, jupyter, ipykernel"
```

## Register venv like kernel jupyter
```
python -m ipykernel install --user --name covid19-venv --display-name "Python (covid-19)"
```

# Validate registering
```
python .\src\validate_venv.py
```

# Directories
```
data-science-machine-learning-for-covid-19-using-Python/
├── data/
├── notebook/                  
├── src/                                 
│   ├── config/
│   │   └── settings.py
│   │
│   ├── evaluation/
│   │   └── evaluator.py
│   │
│   ├── inference/
│   │   └── predictor.py
│   │
│   ├── ingestion/
│   │   └── data_loader.py
│   │
│   ├── persistence/
│   │   └── model_persistence.py
│   │
│   ├── pipeline/
│   │   └── ml_pipeline.py
│   │            
│   ├── preprocessing/
│   │   └── apply_random_undersampling.py
│   │   └── apply_smote_oversampling.py
│   │   └── apply_undersampling.py
│   │   └── feature_selector.py
│   │   └── imbalance_handler.py
│   │
│   ├── processing/
│   │   └── data_cleaner.py
│   │   └── feature_encoder.py
│   │
│   ├── training/
│   │   └── trainer.py
│   │   
│   ├── transformation/
│   │   └── class_weights.py
│   │   └── data_resampling.py
│   │   └── understand_imbalance.py
│   │
│   └── utils/
│       └── exceptions.py
│       └── gitignore_creater.py
│       └── logger.py
│       └── python_environment.py
│       └── reduce_dataset.py
│
├── README.md                  
└── requirements.txt           
```

# Code

- ```src/config/settings.py```
```
from dataclasses import dataclass

@dataclass
class ModelConfig:
    learning_rate: float = 0.2
    n_estimators: int = 200
    max_depth: int = 3
    test_size: float = 0.2
    random_state: int = 1
```

- ```src/evaluation/evaluator.py```
```
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
```

- ```src/inference/predictor.py```
```
import pandas as pd
from typing import Any
from utils.logger import get_logger
from utils.exceptions import MLProjectException

logger = get_logger(__name__)

class Predictor:
    """Responsável por gerar predições."""

    def __init__(self, model: Any):
        self.model = model

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        try:
            logger.info("Gerando previsões")
            return self.model.predict(X_test)
        except Exception as e:
            logger.error("Erro na predição", exc_info=True)
            raise MLProjectException(e)
```

- ```src/ingestion/data_loader.py```
```
"""
data_loader.py

Módulo responsável por carregar datasets localizados na pasta /data
considerando as camadas: raw, clean, processed e storage.
"""

from pathlib import Path
from typing import Optional
import pandas as pd


class DataLoader:
    """
    Classe responsável por localizar e carregar arquivos de dados
    a partir das camadas do data lake local.
    """

    VALID_LAYERS = {"raw", "clean", "processed", "storage"}

    def __init__(self, project_root: Optional[Path] = None) -> None:
        """
        Inicializa o carregador de dados.

        Args:
            project_root (Optional[Path]): Caminho raiz do projeto.
                                           Se None, o caminho é resolvido automaticamente.
        """
        self.project_root: Path = (
            project_root if project_root else self._resolve_project_root()
        )
        self.data_dir: Path = self.project_root / "data"

    def _resolve_project_root(self) -> Path:
        """
        Resolve automaticamente a raiz do projeto assumindo
        que o script está em src/ e os notebooks em notebook/.

        Returns:
            Path: Caminho da raiz do projeto.
        """
        return Path(__file__).resolve().parents[2]

    def load_csv(
        self,
        filename: str,
        layer: str = "raw",
        encoding: str = "utf-8",
        sep: str = ","
    ) -> pd.DataFrame:
        """
        Carrega um arquivo CSV de uma camada específica do data lake.

        Args:
            filename (str): Nome do arquivo CSV.
            layer (str): Camada dos dados (raw, clean, processed, storage).
            encoding (str): Codificação do arquivo.
            sep (str): Separador do CSV.

        Returns:
            pd.DataFrame: DataFrame carregado.
        """
        try:
            if layer not in self.VALID_LAYERS:
                raise ValueError(
                    f"Invalid data layer '{layer}'. "
                    f"Valid layers are: {self.VALID_LAYERS}"
                )

            layer_dir: Path = self.data_dir / layer

            if not layer_dir.exists():
                raise FileNotFoundError(f"Data layer directory not found: {layer_dir}")

            file_path: Path = layer_dir / filename

            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            return pd.read_csv(file_path, encoding=encoding, sep=sep)

        except Exception as error:
            raise RuntimeError(
                f"Error loading CSV file from layer '{layer}': {error}"
            ) from error

```

- ```src/utils/gitignore_creater.py```
```
"""
create_gitignore.py

Cria um arquivo .gitignore na raiz do repositório Git (COVID-19),
listando automaticamente os diretórios abaixo da raiz e adicionando
regras no formato: <diretorio>/data/
"""

from pathlib import Path
from typing import List


class GitRepositoryNotFoundError(Exception):
    """Erro levantado quando a raiz do repositório Git não é encontrada."""
    pass


class GitignoreGenerator:
    """
    Classe responsável por localizar a raiz do repositório Git,
    identificar diretórios de projetos e gerar o arquivo .gitignore.
    """

    def __init__(self, start_path: Path) -> None:
        """
        Inicializa o gerador.

        Args:
            start_path (Path): Caminho inicial para busca da raiz Git.
        """
        self.start_path: Path = start_path
        self.repo_root: Path = self._find_git_root()
        self.gitignore_path: Path = self.repo_root / ".gitignore"

    def _find_git_root(self) -> Path:
        """
        Procura a raiz do repositório Git (diretório que contém .git).

        Returns:
            Path: Caminho da raiz do repositório.

        Raises:
            GitRepositoryNotFoundError: Se a raiz não for encontrada.
        """
        current: Path = self.start_path

        while current != current.parent:
            if (current / ".git").is_dir():
                return current
            current = current.parent

        raise GitRepositoryNotFoundError(
            "Git repository root (.git directory) not found."
        )

    def _list_project_directories(self) -> List[Path]:
        """
        Lista os diretórios imediatamente abaixo da raiz do repositório,
        ignorando pastas ocultas e a pasta .git.

        Returns:
            List[Path]: Lista de diretórios de projeto.
        """
        return [
            item for item in self.repo_root.iterdir()
            if item.is_dir() and not item.name.startswith(".") and item.name != ".git"
        ]

    def _build_ignore_patterns(self) -> List[str]:
        """
        Constrói as regras do .gitignore, incluindo <diretorio>/data/.

        Returns:
            List[str]: Lista de padrões do .gitignore.
        """
        base_patterns: List[str] = [
            "# =========================",
            "# Python",
            "# =========================",
            "__pycache__/",
            "*.py[cod]",
            "*.pyd",
            "*.pyo",
            ".Python",
            "",
            "# =========================",
            "# Virtual environments",
            "# =========================",
            "env/",
            "venv/",
            ".venv/",
            ".env/",
            "",
            "# =========================",
            "# Jupyter Notebook",
            "# =========================",
            "*.ipynb_checkpoints/",
            "",
            "# =========================",
            "# IDEs and editors",
            "# =========================",
            ".vscode/",
            ".idea/",
            "*.code-workspace",
            "",
            "# =========================",
            "# Operating system files",
            "# =========================",
            ".DS_Store",
            "Thumbs.db",
            "desktop.ini",
            "",
            "# =========================",
            "# Logs and reports",
            "# =========================",
            "*.log",
            "logs/",
            "reports/",
            "",
            "# =========================",
            "# Build and distribution",
            "# =========================",
            "build/",
            "dist/",
            "*.egg-info/",
            "",
            "# =========================",
            "# Cache and temporary files",
            "# =========================",
            ".cache/",
            "tmp/",
            "temp/",
            "",
            "# =========================",
            "# Testing and coverage",
            "# =========================",
            ".pytest_cache/",
            ".coverage",
            "htmlcov/",
            "",
            "# =========================",
            "# Project data directories",
            "# =========================",
        ]


        project_dirs: List[Path] = self._list_project_directories()

        data_patterns: List[str] = [
            f"{project_dir.name}/data/"
            for project_dir in project_dirs
        ]

        return base_patterns + data_patterns

    def create_gitignore(self) -> bool:
        """
        Cria o arquivo .gitignore na raiz do repositório.

        Returns:
            bool: True se criado com sucesso, False caso contrário.
        """
        try:
            if self.gitignore_path.exists():
                raise FileExistsError(
                    f".gitignore already exists at {self.gitignore_path}"
                )

            patterns: List[str] = self._build_ignore_patterns()

            with self.gitignore_path.open(mode="w", encoding="utf-8") as file:
                file.write("\n".join(patterns))

            print(
                f".gitignore created at {self.gitignore_path} "
                f"with {len(patterns)} rules."
            )
            return True

        except FileExistsError as error:
            print(f"Warning: {error}")
            return False

        except PermissionError:
            print("Error: Permission denied while creating .gitignore.")
            return False

        except Exception as error:
            print(f"Unexpected error: {error}")
            return False


def main() -> None:
    """
    Função principal do script.
    """
    try:
        script_dir: Path = Path(__file__).resolve().parent
        generator: GitignoreGenerator = GitignoreGenerator(script_dir)
        generator.create_gitignore()

    except GitRepositoryNotFoundError as error:
        print(f"Fatal error: {error}")

    except Exception as error:
        print(f"Unexpected fatal error: {error}")


if __name__ == "__main__":
    main()

```


- ```src/persistence/model/persistence.py```
```
"""
model_persistence.py

Módulo responsável pela persistência de modelos de Machine Learning.

Inclui funcionalidades para:
- Salvar modelos treinados em disco
- Garantir rastreabilidade e reprodutibilidade
- Centralizar tratamento de erros e logging
"""

from pathlib import Path
import pickle
from typing import Any

from utils.logger import get_logger
from utils.exceptions import MLProjectException


logger = get_logger(__name__)


def save_model(
    model: Any,
    filename: str,
    directory: str = "models"
) -> Path:
    """
    Salva um modelo treinado em disco utilizando pickle.

    Args:
        model (Any): Modelo treinado (ex: scikit-learn estimator)
        filename (str): Nome do arquivo de saída (ex: model.sav)
        directory (str): Diretório onde o modelo será salvo

    Returns:
        Path: Caminho completo do arquivo salvo

    Raises:
        MLProjectException: Em caso de falha ao salvar o modelo
    """
    try:
        logger.info("Iniciando salvamento do modelo")

        if model is None:
            raise MLProjectException("Model object is None. Nothing to save.")

        model_dir = Path(directory)
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / filename

        logger.info("Salvando modelo em: %s", model_path)

        with open(model_path, "wb") as file:
            pickle.dump(model, file)

        logger.info("Modelo salvo com sucesso")

        return model_path

    except Exception as error:
        logger.error(
            "Erro ao salvar o modelo em disco",
            exc_info=True
        )
        raise MLProjectException(error) from error

```


- ```src/pipeline/ml_pipeine.py ```
```
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

```

- ````src/prepocessing/apply_random_undersampling.py```
```
"""
apply_random_undersampling.py

Módulo responsável por aplicar RandomUnderSampler
utilizando os dados de treino armazenados no ModelTrainer.

Este módulo:
- NÃO altera o trainer
- NÃO expõe X_train / y_train no notebook
- Segue a arquitetura atual do projeto
"""

from typing import Tuple

import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

from utils.logger import get_logger
from utils.exceptions import MLProjectException


logger = get_logger(__name__)


def apply_random_undersampling_from_trainer(
    trainer,
    random_state: int = 0
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Aplica RandomUnderSampler usando os dados de treino
    armazenados no ModelTrainer.

    Args:
        trainer: Instância treinada de ModelTrainer
        random_state (int): Semente de aleatoriedade

    Returns:
        Tuple[pd.DataFrame, pd.Series]: X e y balanceados
    """
    try:
        logger.info("Iniciando RandomUnderSampler a partir do trainer")

        if not hasattr(trainer, "X_train") or not hasattr(trainer, "y_train"):
            raise MLProjectException(
                "Trainer does not contain training data. "
                "Ensure trainer.train() was executed."
            )

        sampler = RandomUnderSampler(random_state=random_state)

        X_resampled, y_resampled = sampler.fit_resample(
            trainer.X_train,
            trainer.y_train
        )

        logger.info(
            "RandomUnderSampler aplicado | Tamanho antes: %d | depois: %d",
            len(trainer.y_train),
            len(y_resampled)
        )

        return X_resampled, y_resampled

    except Exception as error:
        logger.error(
            "Erro ao aplicar RandomUnderSampler a partir do trainer",
            exc_info=True
        )
        raise MLProjectException(error) from error

```

- ````src/prepocessing/apply_smote_oversampling.py```
```
"""
apply_smote_oversampling.py

Módulo responsável por aplicar técnicas de over-sampling
(SMOTE e ADASYN) utilizando os dados de treino armazenados
no ModelTrainer.

Este módulo:
- NÃO altera o trainer
- NÃO expõe X_train / y_train no notebook
- Segue a arquitetura atual do projeto
"""

from typing import Tuple, Literal

import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN

from utils.logger import get_logger
from utils.exceptions import MLProjectException


logger = get_logger(__name__)


def apply_smote_from_trainer(
    trainer,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Aplica SMOTE usando os dados de treino do ModelTrainer.

    Args:
        trainer: Instância treinada de ModelTrainer
        random_state (int): Semente de aleatoriedade

    Returns:
        Tuple[pd.DataFrame, pd.Series]: X e y balanceados
    """
    return _apply_oversampling(
        trainer=trainer,
        method="smote",
        random_state=random_state
    )


def apply_adasyn_from_trainer(
    trainer,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Aplica ADASYN usando os dados de treino do ModelTrainer.

    Args:
        trainer: Instância treinada de ModelTrainer
        random_state (int): Semente de aleatoriedade

    Returns:
        Tuple[pd.DataFrame, pd.Series]: X e y balanceados
    """
    return _apply_oversampling(
        trainer=trainer,
        method="adasyn",
        random_state=random_state
    )


def _apply_oversampling(
    trainer,
    method: Literal["smote", "adasyn"],
    random_state: int
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Função interna compartilhada para over-sampling.
    """
    try:
        logger.info("Iniciando over-sampling (%s) a partir do trainer", method)

        if not hasattr(trainer, "X_train") or not hasattr(trainer, "y_train"):
            raise MLProjectException(
                "Trainer does not contain training data. "
                "Ensure trainer.train() was executed."
            )

        if method == "smote":
            sampler = SMOTE(random_state=random_state)
        elif method == "adasyn":
            sampler = ADASYN(random_state=random_state)
        else:
            raise MLProjectException(f"Unsupported oversampling method: {method}")

        X_resampled, y_resampled = sampler.fit_resample(
            trainer.X_train,
            trainer.y_train
        )

        logger.info(
            "%s aplicado | Tamanho antes: %d | depois: %d",
            method.upper(),
            len(trainer.y_train),
            len(y_resampled)
        )

        return X_resampled, y_resampled

    except Exception as error:
        logger.error(
            "Erro ao aplicar %s oversampling a partir do trainer",
            method.upper(),
            exc_info=True
        )
        raise MLProjectException(error) from error
    
```

- ````src/prepocessing/apply_undersampling.py```
```
"""
apply_undersampling.py

Módulo responsável por aplicar under-sampling
a partir de um ModelTrainer já treinado.

Este módulo NÃO modifica o trainer,
apenas consome seu estado interno de forma segura.
"""

from typing import Tuple

import pandas as pd

from preprocessing.imbalance_handler import ClusterCentroidsUnderSampler
from utils.exceptions import MLProjectException
from utils.logger import get_logger


logger = get_logger(__name__)


def apply_undersampling_from_trainer(
    trainer
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Aplica under-sampling utilizando os dados de treino
    armazenados no ModelTrainer.

    Args:
        trainer: Instância treinada de ModelTrainer

    Returns:
        Tuple[pd.DataFrame, pd.Series]: X e y balanceados
    """
    try:
        logger.info("Iniciando under-sampling a partir do trainer")

        if not hasattr(trainer, "X_train") or not hasattr(trainer, "y_train"):
            raise MLProjectException(
                "Trainer does not contain training data. "
                "Ensure trainer.train() was executed."
            )

        sampler = ClusterCentroidsUnderSampler()

        X_resampled, y_resampled = sampler.fit_resample(
            X=trainer.X_train,
            y=trainer.y_train
        )

        logger.info("Under-sampling aplicado com sucesso")

        return X_resampled, y_resampled

    except Exception as error:
        logger.error(
            "Erro ao aplicar under-sampling a partir do trainer",
            exc_info=True
        )
        raise MLProjectException(error) from error

```

- ````src/prepocessing/feature_selector.py```
```
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

```

- ````src/prepocessing/imbalance_handler.py```
```
"""
imbalance_handler.py

Módulo responsável por tratar desbalanceamento de classes.
Implementa técnicas de under-sampling usando ClusterCentroids.
"""

from typing import Tuple

import pandas as pd
from imblearn.under_sampling import ClusterCentroids

from utils.logger import get_logger
from utils.exceptions import MLProjectException


logger = get_logger(__name__)


class ClusterCentroidsUnderSampler:
    """
    Aplica under-sampling utilizando o algoritmo ClusterCentroids.
    """

    def __init__(
        self,
        random_state: int = 42,
        sampling_strategy: str = "auto"
    ) -> None:
        self.random_state = random_state
        self.sampling_strategy = sampling_strategy
        self._sampler = ClusterCentroids(
            random_state=self.random_state,
            sampling_strategy=self.sampling_strategy
        )

    def fit_resample(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Executa o under-sampling nos dados.

        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Dados balanceados
        """
        try:
            logger.info("Aplicando ClusterCentroids under-sampling")

            if not isinstance(X, pd.DataFrame):
                raise MLProjectException("X must be a pandas DataFrame")

            if not isinstance(y, pd.Series):
                raise MLProjectException("y must be a pandas Series")

            X_resampled, y_resampled = self._sampler.fit_resample(X, y)

            logger.info(
                "Under-sampling concluído | Tamanho antes: %d | depois: %d",
                len(y),
                len(y_resampled)
            )

            return X_resampled, y_resampled

        except Exception as error:
            logger.error(
                "Erro ao aplicar ClusterCentroids under-sampling",
                exc_info=True
            )
            raise MLProjectException(error) from error

```

- ```src/procesing/data_cleaner.py```
```
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

```

- ```src/procesing/feature_encoder.py```
```
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

```

- ```src/training/trainer.py```
```
"""
trainer.py

Módulo responsável pelo treinamento do modelo de Machine Learning.

Responsabilidades:
- Validação do target
- Split estratificado
- Logging detalhado
- Tratamento de exceções
- Armazenamento de estado (X_train, X_test, y_train, y_test)
- Inicialização e treinamento do modelo
"""

from typing import Tuple
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

from utils.logger import get_logger
from utils.exceptions import MLProjectException
from config.settings import ModelConfig


logger = get_logger(__name__)


class ModelTrainer:
    """
    Classe responsável pela etapa de treinamento do modelo.
    Encapsula todo o estado necessário para avaliação posterior.
    """

    def __init__(self, config: ModelConfig) -> None:
        """
        Inicializa o treinador com as configurações do modelo.

        Args:
            config (ModelConfig): Configurações do modelo
        """
        self.config: ModelConfig = config
        self.model: GradientBoostingClassifier | None = None

        # Estado do treino (exposto para avaliação)
        self.X_train: pd.DataFrame | None = None
        self.X_test: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None
        self.y_test: pd.Series | None = None

    # =========================
    # API pública
    # =========================
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Executa o fluxo completo de treinamento do modelo.

        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
                X_train, X_test, y_train, y_test
        """
        try:
            logger.info("Iniciando processo de treinamento")

            self._validate_inputs(X, y)
            self._validate_target(y)

            self._split_data(X, y)
            self._log_class_distribution()

            self._initialize_model()
            self._fit_model()

            logger.info("Treinamento concluído com sucesso")

            return self.X_train, self.X_test, self.y_train, self.y_test

        except Exception as error:
            logger.error(
                "Erro durante o treinamento do modelo",
                exc_info=True
            )
            raise MLProjectException(error) from error

    # =========================
    # Métodos internos
    # =========================
    def _validate_inputs(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Valida os tipos de entrada.
        """
        if not isinstance(X, pd.DataFrame):
            raise MLProjectException("X must be a pandas DataFrame.")

        if not isinstance(y, pd.Series):
            raise MLProjectException("y must be a pandas Series.")

        if X.empty or y.empty:
            raise MLProjectException("X and y must not be empty.")

        logger.info("Entradas validadas com sucesso")

    def _validate_target(self, y: pd.Series) -> None:
        """
        Valida se o target possui pelo menos duas classes.
        """
        unique_classes = y.dropna().unique()

        if len(unique_classes) < 2:
            raise MLProjectException(
                "Target variable has less than 2 classes. "
                "Classification requires at least two classes."
            )

        logger.info(
            "Target validado | Classes encontradas: %s",
            unique_classes
        )

    def _split_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Realiza o split estratificado e armazena o estado internamente.
        """
        logger.info("Realizando split estratificado de treino e teste")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )

    def _log_class_distribution(self) -> None:
        """
        Loga a distribuição de classes nos conjuntos de treino e teste.
        """
        logger.info(
            "Distribuição de classes no treino: %s",
            Counter(self.y_train)
        )
        logger.info(
            "Distribuição de classes no teste: %s",
            Counter(self.y_test)
        )

    def _initialize_model(self) -> None:
        """
        Inicializa o modelo de Machine Learning.
        """
        logger.info("Inicializando GradientBoostingClassifier")

        self.model = GradientBoostingClassifier(
            learning_rate=self.config.learning_rate,
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth
        )

    def _fit_model(self) -> None:
        """
        Executa o treinamento do modelo.
        """
        if self.model is None:
            raise MLProjectException("Model has not been initialized.")

        logger.info("Treinando modelo")
        self.model.fit(self.X_train, self.y_train)
```

-```src/transformation/class_weights.py```
```
from sklearn.ensemble import RandomForestClassifier

def scikit_class_weight():
    '''
    Use pesos de classe (Class Weights)
    Muito eficaz quando não dá para alterar o dataset.
    Exemplo (Scikit-learn)
    '''
    model = RandomForestClassifier(class_weight='balanced')

def keras_class_weight():
    '''
    Use pesos de classe (Class Weights)
    Muito eficaz quando não dá para alterar o dataset.
    Exemplo (Deep Learning – Keras)
    '''
    class_weight = {0: 6, 1: 1, 2: 1}


```

-```src/transformation/data_resampling.py```
```
from imblearn.over_sampling import SMOTE

def resampling_values():
    '''
    Reamostragem dos dados
    Oversampling (classe minoritária)
    - Random Oversampling
    - SMOTE
    - ADASYN
    '''

    X_res, y_res = SMOTE().fit_resample(X, y)
```

-```src/utils/reduce_dataset.py```
```
"""
reduce_dataset.py

Reduz o tamanho de um dataset CSV localizado em data/raw/,
salva a versão reduzida em data/processed/ e copia o
dataset final para data/storage/.
"""

from pathlib import Path
import logging
import shutil
import pandas as pd


# =========================
# Logging configuration
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


class DatasetStorageManager:
    """
    Classe responsável por gerenciar o armazenamento final de datasets.
    """

    def __init__(self, storage_dir: Path) -> None:
        """
        Inicializa o gerenciador de storage.

        Args:
            storage_dir (Path): Diretório de armazenamento final.
        """
        self.storage_dir: Path = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def copy_to_storage(self, file_path: Path) -> Path:
        """
        Copia um arquivo para a pasta de storage.

        Args:
            file_path (Path): Caminho do arquivo a ser copiado.

        Returns:
            Path: Caminho do arquivo copiado no storage.
        """
        try:
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            destination: Path = self.storage_dir / file_path.name
            shutil.copy2(file_path, destination)

            logger.info(
                "Dataset copied to storage: %s",
                destination
            )

            return destination

        except Exception as error:
            logger.error(
                "Error copying dataset to storage: %s",
                error,
                exc_info=True
            )
            raise RuntimeError("Dataset storage copy failed.") from error


class DatasetReducer:
    """
    Classe responsável por reduzir o tamanho de datasets CSV
    respeitando um limite máximo de tamanho em MB.
    """

    def __init__(
        self,
        project_root: Path,
        input_filename: str,
        output_filename: str,
        max_size_mb: int = 90,
        random_state: int = 42
    ) -> None:
        self.project_root: Path = project_root
        self.raw_dir: Path = self.project_root / "data" / "raw"
        self.processed_dir: Path = self.project_root / "data" / "processed"
        self.storage_dir: Path = self.project_root / "data" / "storage"

        self.input_file: Path = self.raw_dir / input_filename
        self.output_file: Path = self.processed_dir / output_filename

        self.max_size_mb: int = max_size_mb
        self.random_state: int = random_state

        self.storage_manager = DatasetStorageManager(self.storage_dir)

    @staticmethod
    def _get_file_size_mb(file_path: Path) -> float:
        """Retorna o tamanho do arquivo em MB."""
        return file_path.stat().st_size / (1024 * 1024)

    def reduce(self) -> None:
        """Executa a redução do dataset e copia para storage."""
        try:
            if not self.input_file.exists():
                raise FileNotFoundError(
                    f"Input dataset not found: {self.input_file}"
                )

            self.processed_dir.mkdir(parents=True, exist_ok=True)

            original_size = self._get_file_size_mb(self.input_file)
            logger.info("Original dataset size: %.2f MB", original_size)

            if original_size <= self.max_size_mb:
                logger.info(
                    "Dataset already within size limit (%d MB). Skipping reduction.",
                    self.max_size_mb
                )
                final_file = self.input_file
            else:
                reduction_ratio = self.max_size_mb / original_size
                logger.info(
                    "Applying sampling ratio: %.4f to fit within %d MB",
                    reduction_ratio,
                    self.max_size_mb
                )

                df = pd.read_csv(self.input_file)

                reduced_df = df.sample(
                    frac=reduction_ratio,
                    random_state=self.random_state
                )

                reduced_df.to_csv(self.output_file, index=False)
                final_file = self.output_file

                final_size = self._get_file_size_mb(final_file)
                logger.info(
                    "Reduced dataset saved at: %s (%.2f MB)",
                    final_file,
                    final_size
                )

            # Copia dataset final para storage
            self.storage_manager.copy_to_storage(final_file)

        except Exception as error:
            logger.error(
                "Error during dataset reduction pipeline: %s",
                error,
                exc_info=True
            )
            raise RuntimeError("Dataset reduction pipeline failed.") from error


def main() -> None:
    """Função principal."""
    try:
        # src/transformation/reduce_dataset.py → src → project root
        project_root: Path = Path(__file__).resolve().parents[2]

        reducer = DatasetReducer(
            project_root=project_root,
            input_filename="corona_tested_individuals_ver_0083.english.csv",
            output_filename="corona_tested_individuals_reduced.csv",
            max_size_mb=90
        )

        reducer.reduce()

    except Exception as error:
        logger.critical(
            "Fatal error during dataset pipeline execution: %s",
            error,
            exc_info=True
        )


if __name__ == "__main__":
    main()
```

-```src/transformation/understand_imbalance.py```
```
import pandas as pd

def count_values():
    '''
    Entenda o desbalanceamento antes de modelar:
    - Antes de qualquer coisa:
    - Analise a distribuição das classes
    - Identifique classe majoritária e minoritária
    - Avalie a criticidade do erro (ex: falso negativo)
    '''
    
    return df['label'].value_counts(normalize=True)
```

-```src/utils/exceptions.py```
```
class MLProjectException(Exception):
    """Exceção base do projeto de Machine Learning."""
    pass

```

-```src/utils/logger.py```
```
import logging
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(LOG_DIR / "app.log")
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)

    return logger

```

-```src/utils/python_environment.py```
```
"""
python_environment_checker.py

Script responsável por identificar qual interpretador Python
está sendo utilizado no ambiente atual (terminal ou Jupyter Notebook).
"""

import sys
from typing import Optional


class PythonEnvironmentChecker:
    """
    Classe responsável por verificar informações do ambiente Python.
    """

    def __init__(self) -> None:
        """
        Inicializa o verificador de ambiente.
        """
        pass

    def get_python_executable(self) -> Optional[str]:
        """
        Obtém o caminho completo do executável Python em uso.

        Returns:
            Optional[str]: Caminho do executável Python ou None em caso de erro.
        """
        try:
            executable_path: str = sys.executable

            if not executable_path:
                raise ValueError("Python executable path is empty.")

            return executable_path

        except Exception as error:
            print(f"Error while retrieving Python executable: {error}")
            return None


def print_python_executable() -> None:
    """
    Função responsável por imprimir o executável Python ativo.
    """
    try:
        checker: PythonEnvironmentChecker = PythonEnvironmentChecker()
        executable: Optional[str] = checker.get_python_executable()

        if executable:
            print(f"Active Python executable:\n{executable}")
        else:
            print("Unable to determine the active Python executable.")

    except Exception as error:
        print(f"Unexpected error: {error}")


if __name__ == "__main__":
    print_python_executable()

```
# Data Science & Machine Learning for COVID-19 using Python

### Classifying COVID-19 Patients Using Symptoms

This project addresses a medical classification problem focused on identifying COVID-19 cases based on clinical data.

#### Problem Statement
The objective is to classify patient records into the following categories:
- **COVID-19** (219 samples)
- **Pneumonia** (1,345 samples)
- **Normal** (1,341 samples)

The dataset presents **class imbalance**, with COVID-19 being the minority and most critical class.

#### Challenges
- **Class imbalance** can lead to biased models that favor majority classes.
- **Accuracy alone is misleading**, as high overall accuracy may hide poor detection of COVID-19 cases.
- **False negatives** are particularly harmful in healthcare contexts.

#### Approach
To mitigate these issues, the project adopts best practices for imbalanced classification:
- Analysis of class distribution before modeling
- Use of appropriate evaluation metrics beyond accuracy:
  - Recall, Precision, F1-score
  - Confusion Matrix
  - ROC-AUC and Precision-Recall AUC
- Application of data resampling techniques:
  - Undersampling and Oversampling (SMOTE, ADASYN)
- Consideration of class weighting and threshold tuning
- Stratified validation to preserve class proportions

This approach ensures a more reliable and clinically meaningful model evaluation.


### Coding

#### Update the PIP
```
python.exe -m pip install --upgrade pip
```

#### Create gitignore file
```
python .\src\utils\gitignore_creater.py
```

#### Reduce Dataset
```
src/utils/reduce_dataset.py
```

#### Create environment venv
```
python -m venv .venv
```

#### Active environment venv
```
.venv\Scripts\Activate.ps1
```

#### Install dependencies
```
pip install -r requirements.txt
```

#### Validate dependencies
```
python -c "import numpy, pandas, matplotlib, sklearn, imblearn, jupyter, ipykernel"
```

#### Register venv like kernel jupyter
```
python -m ipykernel install --user --name covid19-venv --display-name "Python (covid-19)"
```

#### Validate registering
```
python .\src\validate_venv.py
```

### Directories
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
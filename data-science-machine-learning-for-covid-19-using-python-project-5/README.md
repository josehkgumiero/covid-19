# Chest X-Ray Classification – COVID-19 Detection with Deep Learning

This project implements a complete **Deep Learning pipeline** for classifying chest X-ray images into three categories:

- COVID-19
- Normal
- Pneumonia

The solution was developed using **Python and TensorFlow/Keras**, following best practices in machine learning workflow, dataset organization, model training, evaluation, and optimization.

---

# Project Overview

The goal of this project is to build a Convolutional Neural Network (CNN) capable of analyzing chest X-ray images and predicting whether a patient:

- Has COVID-19
- Has Pneumonia
- Is Normal

This project demonstrates:

- End-to-end ML pipeline
- Image preprocessing
- Deep learning modeling
- Fine-tuning techniques
- Performance evaluation
- Production-ready dataset handling

---

# Project Structure
```
data-science-machine-learning-for-covid-19-using-python-project-5/
│
├── data/
│ ├── raw/ # Full original dataset
│ │ ├── covid/
│ │ ├── normal/
│ │ └── pneumonia/
│ │
│ ├── github_dataset/ # Reduced dataset for GitHub version
│ │ ├── covid/
│ │ ├── normal/
│ │ └── pneumonia/
│ │
│ ├── processed/ # Processed or transformed data
│ └── storage/ # Backup / overflow dataset
│
├── Chest_X_ray_Classification_v2.ipynb # Main ML notebook
├── dataset_size_controller.py # Dataset size control script
├── requirements.txt
└── README.md
```

---

# Machine Learning Pipeline

The notebook implements a structured ML workflow:

## 1 Data Loading

- Images are loaded from the dataset directory
- Organized by class folders
- Automatically labeled based on folder name

---

## 2 Data Preprocessing

- Image resizing
- Normalization (pixel scaling)
- Data splitting (train / validation)
- Batch generation
- Optional class balancing

---

## 3 Model Architecture

The model uses a Convolutional Neural Network (CNN), potentially including:

- Convolutional layers
- Pooling layers
- Fully connected layers
- Softmax output layer (multi-class classification)

Advanced techniques used:

- Fine-tuning (transfer learning if applied)
- EarlyStopping callback
- ModelCheckpoint
- Dropout (regularization)

---

## 4 Model Training

During training:

- Loss and accuracy are tracked
- Training and validation curves are plotted
- Early stopping prevents overfitting
- Best model weights are saved

---

## 5 Model Evaluation

Evaluation includes:

- Accuracy score
- Confusion matrix
- Classification report (precision, recall, F1-score)
- Loss and accuracy visualization

---

# Dataset Handling Strategy

Because GitHub has size limitations, this project includes:

### `dataset_size_controller.py`

This script:

- Preserves the full dataset inside `data/raw`
- Creates a balanced reduced dataset in `data/github_dataset`
- Ensures repository size compliance
- Maintains reproducibility

This prevents data loss while keeping the repository lightweight.

---

# How to Run the Project

## 1 Create Virtual Environment

```
python -m venv .venv
```
```
source .venv/bin/activate   
```

```
.venv\Scripts\activate      
```

### 2 Install Dependencies

```
pip install -r requirements.txt
```


## 3 Main libraries used:

numpy

tensorflow

matplotlib

scikit-learn

## 4 Open Notebook
jupyter notebook
Open:
```
Chest_X_ray_Classification_v2.ipynb
Run all cells sequentially.
```

# Model Output
The model predicts one of the following classes:

COVID-19

Pneumonia

Normal

Example output:

Prediction: COVID-19
Confidence: 92%

# Technologies Used
Python 3.10+

TensorFlow / Keras

NumPy

Matplotlib

Scikit-Learn

Jupyter Notebook

# Learning Objectives
This project demonstrates:

Deep learning for medical imaging

CNN-based image classification

Model regularization techniques

Fine-tuning pre-trained models

Data engineering for ML

Dataset size management for GitHub

Structured ML project organization

# Disclaimer
This project is for educational purposes only.
It is not intended for medical diagnosis or clinical use.

# Author
Developed as part of a Machine Learning portfolio project using Deep Learning and Python.


































# Installation dependencies
1. Create a virtual environment
```
python -m venv .venv
```

2. Activate the virtual environment
```
.venv\Scripts\Activate.ps1
```

3. Install dependencies
```
pip install -r requirements.txt
```

#### Register venv like kernel jupyter
```
python -m ipykernel install --user --name covid19-venv-python-3-10 --display-name "Python 3.10 (covid-19)"
```
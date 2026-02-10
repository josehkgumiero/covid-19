# Tweet Classification – COVID-19 Tweets
# Overview

Since the outbreak of the COVID-19 pandemic, social media platforms have been flooded with discussions related to the virus, its impacts, and global responses.
Between January 2020 and the following months, more than 600 million tweets referenced COVID-19-related topics worldwide.

This project aims to build a robust Natural Language Processing (NLP) pipeline to automatically classify tweets as:

COVID-19 related (1)

Non-COVID-19 related (0)

The entire solution was designed with clean architecture principles, object-oriented programming (OOP), and reproducibility in mind, moving beyond exploratory notebooks into a production-ready structure.

# Problem Statement

Can we build a machine learning model capable of identifying whether a tweet is related to COVID-19 even when explicit keywords are removed?

How can we ensure consistency between tokenization, embeddings, and model training?

How do we design an NLP pipeline that is modular, reusable, and scalable?

# Dataset Description

The dataset used in this project was originally released as part of a Hackathon organized by Zindi.Africa, sponsored by Microsoft.

Key characteristics:

Tweets are labeled as:

1 → COVID-19 related

0 → Not COVID-19 related

All explicit COVID-related keywords were removed, including:

corona

coronavirus

covid

covid19

covid-19

sarscov2

19

This makes the task significantly more challenging, as the model must rely on contextual and semantic patterns, not keywords.

# Solution Architecture

The solution follows a fully modular NLP pipeline, implemented with object-oriented design.

# High-level pipeline
```
Raw Tweets
   ↓
Text Preprocessing
   ↓
Tokenization & Padding
   ↓
Word2Vec Embedding Matrix
   ↓
LSTM Neural Network
   ↓
Training & Evaluation
   ↓
Visualization of Metrics
```

```
🗂️ Project Structure
src/
├── config/
│   └── settings.py                # Global constants (embedding dim, sequence length)
│
├── utils/
│   └── logger.py                  # Centralized logging
│
├── nlp/
│   ├── text_preprocessor.py       # Tokenization, cleaning, stopwords
│   ├── word2vec_trainer.py        # Word2Vec model training
│   ├── embedding_loader.py        # Load embeddings from disk
│   ├── embedding_matrix_builder.py# Align embeddings with tokenizer
│   └── sequence_builder.py        # Tokenizer, padding, train/test split
│
├── models/
│   └── lstm_embedding_classifier.py  # LSTM model with pretrained embeddings
│
├── training/
│   ├── trainer.py                 # Training abstraction
│   └── evaluator.py               # Model evaluation
│
└── visualization/
    └── training_plotter.py        # Accuracy & loss plots
```

# Model Architecture

Embedding Layer

Initialized with pretrained Word2Vec embeddings

Non-trainable to preserve semantic structure

LSTM Layer

Captures sequential and contextual dependencies

Dense Output Layer

Sigmoid activation for binary classification

Loss Function: Binary Crossentropy
Optimizer: Adam
Metric: Accuracy

# Training & Evaluation

Data is shuffled with a fixed random seed for reproducibility

Train/Test split handled centrally by the SequenceBuilder

Training history includes:

Training accuracy & loss

Validation accuracy & loss

Visualization is handled via a dedicated plotting module

# Key Technical Highlights

✅ Single source of truth for the tokenizer (prevents embedding mismatch)

✅ Explicit handling of embedding–tokenizer alignment

✅ Defensive programming with validation checks and logging

✅ Fully decoupled notebook and core logic

✅ Ready for extension (CNN, GRU, Transformers, etc.)

# How to Run

Clone the repository

Create and activate a virtual environment

Install dependencies

Run the notebook:

01_pipeline_refatorado.ipynb


The notebook executes the entire pipeline end-to-end:

Data preparation

Model training

Evaluation

Visualization

# Results

Despite the removal of all explicit COVID-19 keywords, the model is able to learn latent semantic patterns and distinguish COVID-related tweets with strong performance, demonstrating the effectiveness of contextual embeddings combined with LSTM networks.

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
python -m ipykernel install --user --name covid19-venv-python-3-11 --display-name "Python 3.11 (covid-19)"
```

# Future Improvements

Hyperparameter optimization

Use of pretrained embeddings (GloVe / FastText)

Transformer-based architectures (BERT, DistilBERT)

Model explainability (SHAP / LIME)

Experiment tracking (MLflow)

# License

This project is intended for educational and research purposes.
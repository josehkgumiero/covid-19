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
python .\src\ingestion\gitignore_creater.py
```

## Reduce Dataset
```
src/transformation/reduce_dataset.py
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
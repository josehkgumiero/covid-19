# Academic Report  
## Machine Learning Pipeline for COVID-19 Symptom-Based Classification

---

## 1. Introduction

The COVID-19 pandemic highlighted the importance of data-driven approaches for rapid and reliable decision-making in healthcare. Machine Learning (ML) techniques provide powerful tools to analyze clinical and epidemiological data, enabling predictive modeling that can assist in early diagnosis and resource allocation.

This project presents an end-to-end Machine Learning pipeline designed to classify COVID-19 test outcomes based on patient symptoms and demographic information. Beyond model training, the project emphasizes **robust engineering practices**, **correct evaluation strategies**, and **proper handling of class imbalance**, which are critical for real-world healthcare applications.

---

## 2. Dataset Description

The dataset consists of individual-level COVID-19 test records, containing both **symptom indicators** and **demographic attributes**. Each record represents a patient tested for COVID-19.

### 2.1 Target Variable

- **corona_result**  
  Binary classification target:
  - `0`: Negative
  - `1`: Positive

### 2.2 Feature Variables

| Feature | Description | Type |
|------|-----------|------|
| cough | Presence of cough symptom | Binary |
| fever | Presence of fever symptom | Binary |
| sore_throat | Presence of sore throat | Binary |
| shortness_of_breath | Breathing difficulty | Binary |
| head_ache | Presence of headache | Binary |
| age_60_and_above | Age group (≥ 60 years) | Binary |
| gender | Biological sex (male/female) | Binary |
| contact_with_confirmed | Contact with confirmed COVID-19 case | Binary |

All categorical variables were encoded into numerical binary values to ensure compatibility with machine learning algorithms.

---

## 3. Data Preprocessing

### 3.1 Data Cleaning

Missing values were removed from the dataset to ensure consistency and avoid biased model behavior. Dataset dimensions before and after cleaning were logged to maintain traceability.

### 3.2 Feature Encoding

Categorical variables were transformed using vectorized encoding strategies, avoiding row-wise operations for better performance and reproducibility.

---

## 4. Class Imbalance Analysis

The dataset presents a **class imbalance**, where negative test results significantly outnumber positive cases. This scenario is common in real-world medical datasets and introduces several risks:

- Bias toward the majority class
- Misleading accuracy metrics
- Increased false negatives for the critical minority class

Given the healthcare context, **false negatives (COVID-positive classified as negative)** are particularly harmful.

---

## 5. Strategies for Handling Class Imbalance

To mitigate imbalance-related issues, multiple strategies were considered:

- **Random Undersampling**
- **SMOTE (Synthetic Minority Oversampling Technique)**
- **ADASYN (Adaptive Synthetic Sampling)**
- **Class weighting (theoretical discussion)**

In this pipeline, **Random Undersampling** was applied to the training set to create a balanced dataset while preserving a clean evaluation protocol.

---

## 6. Machine Learning Model

### 6.1 Algorithm Selection

The chosen algorithm was **Gradient Boosting Classifier**, which offers:

- Strong performance on tabular data
- Ability to capture non-linear feature interactions
- Robustness against feature noise
- Good bias–variance tradeoff

### 6.2 Model Configuration

Hyperparameters were centralized in a configuration object:

- Learning rate
- Number of estimators
- Maximum tree depth
- Train-test split ratio
- Random seed for reproducibility

---

## 7. Training and Validation Strategy

- **Stratified train-test split** was applied to preserve class proportions.
- Model state (X_train, X_test, y_train, y_test) was stored for reproducible evaluation.
- Logging ensured full auditability of the training process.

---

## 8. Evaluation Metrics

Due to class imbalance, evaluation relied on metrics beyond accuracy:

- **Accuracy** – Overall correctness
- **Precision** – Reliability of positive predictions
- **Recall (Sensitivity)** – Ability to detect positive cases
- **Confusion Matrix** – Error distribution analysis
- **ROC Curve & AUC** – Discriminative power across thresholds
- **Classification Report** – Per-class performance summary

### 8.1 Metric Interpretation

- High recall is prioritized to minimize false negatives.
- Precision-recall trade-offs were analyzed to understand model behavior.
- ROC-AUC provides a threshold-independent evaluation.

---

## 9. Results and Visual Analysis

### 9.1 Confusion Matrix

The confusion matrix highlights:
- True Positives (correct COVID-19 detection)
- False Negatives (missed positive cases)
- False Positives
- True Negatives

This visualization provides direct insight into clinical risk.

### 9.2 ROC Curve

The ROC curve demonstrates the model’s ability to separate positive and negative cases. A higher AUC indicates stronger discriminative performance.

---

## 10. Feature Influence and Correlation Analysis

Although Gradient Boosting does not expose simple linear coefficients, feature importance analysis and correlation inspection reveal:

- **Contact with confirmed cases** strongly correlates with positive outcomes
- **Fever and cough** are significant predictors
- **Age ≥ 60** increases risk probability
- Some symptoms exhibit correlated behavior, reinforcing clinical consistency

Correlation analysis confirms that symptom co-occurrence aligns with medical expectations, validating dataset reliability.

---

## 11. Model Persistence

The final trained model was serialized and stored on disk, enabling:

- Reproducible inference
- Deployment readiness
- Experiment versioning

Model persistence ensures separation between training and inference environments.

---

## 12. Engineering and Architecture Considerations

This project adopts software engineering best practices applied to ML:

- Modular architecture
- Object-oriented design
- Centralized logging
- Explicit exception handling
- Separation between notebooks and production code
- Reusable pipeline components

Such practices are essential for scalable and maintainable ML systems.

---

## 13. Conclusion

This work demonstrates that effective Machine Learning in healthcare requires more than algorithm selection. Proper **data preprocessing**, **imbalance handling**, **metric selection**, and **engineering discipline** are critical for producing reliable and trustworthy models.

The proposed pipeline serves both as a learning reference and as a foundation for real-world Machine Learning applications, emphasizing reproducibility, interpretability, and responsible evaluation.

---

## 14. Future Work

Potential extensions include:
- Cross-validation strategies
- Cost-sensitive learning
- Threshold optimization
- Model explainability with SHAP
- Integration with MLOps platforms
- Deployment as an inference service

---

## References

- Scikit-learn documentation  
- Imbalanced-learn documentation  
- Machine Learning for Healthcare literature  
- WHO COVID-19 data analysis guidelines

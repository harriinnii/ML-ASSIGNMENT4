# Ensemble Prediction and Decision Tree Model Evaluation

## Aim

To build classifiers such as **Decision Tree, AdaBoost, Gradient
Boosting, XGBoost, Random Forest, and Stacked Models (SVM + Naïve
Bayes + Decision Tree)** and evaluate their performance through **5-Fold
Cross-Validation** and hyperparameter tuning.

## Objectives

-   Perform preprocessing on the **Breast Cancer Wisconsin Dataset**.\
-   Train and tune multiple ensemble classifiers.\
-   Evaluate models using **Accuracy, Precision, Recall, F1 Score, ROC
    Curve, and Confusion Matrix**.\
-   Compare base Decision Tree vs Ensemble methods.

## Dataset

-   **Breast Cancer Wisconsin Dataset** (loaded from
    `sklearn.datasets.load_breast_cancer`).\
-   Features: 30 numeric input features.\
-   Target: Binary classification (Malignant vs Benign).

## Models Implemented

1.  Decision Tree (with hyperparameter tuning)\
2.  AdaBoost Classifier\
3.  Gradient Boosting Classifier\
4.  XGBoost Classifier\
5.  Random Forest Classifier\
6.  Stacked Ensemble (SVM + Naïve Bayes + Decision Tree, Logistic
    Regression as meta-learner)

## Libraries Used

-   **numpy, pandas** → Data handling\
-   **matplotlib, seaborn** → Visualization\
-   **scikit-learn** → Models (Decision Tree, AdaBoost, Gradient
    Boosting, Random Forest, Stacking), metrics, cross-validation\
-   **xgboost** → XGBoost classifier

## How to Run

1.  Clone or download this repository.\

2.  Install required libraries:

    ``` bash
    pip install numpy pandas matplotlib seaborn scikit-learn xgboost
    ```

3.  Run the Python script:

    ``` bash
    python ensemble_evaluation.py
    ```

4.  The script will:

    -   Preprocess the dataset\
    -   Train all models with hyperparameter tuning\
    -   Display evaluation metrics, confusion matrices, and ROC curves\
    -   Print 5-fold cross-validation results

## Results

### Test Set Performance

  Model               Accuracy     F1 Score
  ------------------- ------------ ------------
  Decision Tree       91.23%       92.75%
  AdaBoost            95.61%       96.60%
  Gradient Boosting   95.61%       96.60%
  XGBoost             94.74%       95.89%
  Random Forest       95.61%       96.55%
  Stacked Ensemble    **97.37%**   **97.93%**

### 5-Fold Cross-Validation Average Accuracy

-   Decision Tree: 93.67%\
-   AdaBoost: **97.72%**\
-   Gradient Boosting: 95.80%\
-   XGBoost: 96.84%\
-   Random Forest: 96.66%\
-   Stacked Ensemble: 95.96%

### Inference

-   **Stacked Ensemble** achieved the best test accuracy (97.37%).\
-   **AdaBoost** performed best in cross-validation (97.72%).\
-   All ensemble methods outperformed the standalone Decision Tree.

## Learning Outcomes

-   Learned model-by-model evaluation with classification metrics.\
-   Understood hyperparameter tuning and performance trade-offs.\
-   Practiced visualizing results via confusion matrices and ROC
    curves.\
-   Verified that ensemble methods outperform standalone classifiers.\
-   Gained insight into why stacking often generalizes best.

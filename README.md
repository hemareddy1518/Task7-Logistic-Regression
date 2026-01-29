# Titanic Survival Prediction using Logistic Regression

## 1. Objective

The goal of this project is to build a **binary classification model** using **Logistic Regression** to predict whether a passenger survived the Titanic disaster.
This project demonstrates the complete **machine learning workflow**, including data preprocessing, feature engineering, model training, evaluation, and performance visualization.

---

## 2. Dataset Description

The dataset used is the **Titanic dataset** loaded via Seaborn:

```python
import seaborn as sns
df = sns.load_dataset("titanic")
```

* **Target Variable:**

  * `survived`: 0 = Not Survived, 1 = Survived

* **Selected Features for Prediction:**

  * `pclass` (Passenger class)
  * `sex`
  * `age`
  * `sibsp` (Siblings/Spouses aboard)
  * `parch` (Parents/Children aboard)
  * `fare`
  * `embarked` (Boarding port)

> These features were selected because they strongly influence survival chances.

---

## 3. Data Preprocessing

**Steps performed to prepare the dataset for modeling:**

1. **Missing Values Handling**

   * `age`: Filled missing values using the median.
   * `embarked`: Filled missing values using the mode (most frequent value).

2. **Feature Encoding**

   * Categorical features (`sex`, `embarked`, `pclass`) converted to numeric using **OneHotEncoding**.

3. **Feature Scaling**

   * Numerical features (`age`, `fare`, `sibsp`, `parch`) scaled using **StandardScaler** for stable model training.

---

## 4. Model Training

* A **Logistic Regression** model was trained using an **ML pipeline** that integrates preprocessing and training.
* Dataset split:

  * **80% Training Set**
  * **20% Testing Set**
* **Stratified split** applied to maintain class distribution in both sets.

---

## 5. Model Evaluation

The model was evaluated using multiple metrics:

* **Accuracy:** Overall correct predictions
* **Precision:** Correctly predicted survivors out of all predicted survivors
* **Recall:** Correctly predicted survivors out of all actual survivors
* **F1-score:** Balance between precision and recall
* **Confusion Matrix:** Shows TP, TN, FP, FN for detailed classification

Additional evaluation metrics:

* **ROC Curve:** Plots True Positive Rate vs False Positive Rate
* **AUC Score:** Measures overall model discrimination ability (higher is better)

---

## 6. Visual Outputs

The following visualizations were generated and saved:

* **Confusion Matrix:** `confusion_matrix.png`
* **ROC Curve with AUC:** `roc_curve.png`

> These visualizations provide a deeper understanding of model performance beyond accuracy.

---

## 7. Conclusion

This project successfully demonstrates the application of **Logistic Regression** for **binary classification**.
The model was trained with proper preprocessing, including missing value treatment, encoding, and scaling.
Evaluation was done using classification metrics and visual tools such as confusion matrix and ROC-AUC.

---

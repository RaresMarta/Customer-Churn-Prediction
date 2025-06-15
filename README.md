# 🏦 Customer Churn Prediction
![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange)


This project tackles the problem of predicting **customer churn** for a financial institution. The goal is to build accurate machine learning models that estimate the likelihood of a customer exiting the bank (i.e., `Exited = 1`) using demographic, financial, and behavioral data.

---

## 📊 Problem Context

Customer churn refers to the event where a client ceases their relationship with the bank. Proactively identifying at-risk customers enables:

- Retention strategies
- Targeted marketing
- Better resource allocation

---

## 🧠 Objective

- Build a binary classification model to estimate the probability of churn.
- Use `train.csv` to train and validate models.
- Predict probabilities on `test.csv`.
- Output a `submission.csv` containing:  
  ```csv
  id,Exited
  15000,0.327
  15001,0.678
  ```
- Evaluation Metric: **ROC AUC**

---

## 📁 Dataset Description

| File               | Description                                 |
|--------------------|---------------------------------------------|
| `train.csv`        | 15,000 customers with churn label (`Exited`) |
| `test.csv`         | 10,000 customers without `Exited`            |
| `sample_submission.csv` | Format required for final predictions       |

### 🚩 Features:

- `CustomerId`, `Surname`: Identifiers (dropped in modeling)
- `CreditScore`, `Age`, `Balance`, `EstimatedSalary`: Numeric
- `Geography`, `Gender`: Categorical
- `Tenure`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`: Behavioral
- `Exited`: Target (1 = churned, 0 = retained)

---

## 🛠️ Preprocessing Strategies

Three preprocessing pipelines are used:

1. **Unscaled**: Raw numeric features with one-hot encoded categoricals.
2. **Scaled**: StandardScaler on numeric features.
3. **Linear**: Binned numeric features + one-hot encoding (best for Logistic Regression).

---

## 🔍 Models Trained

| Model                | Notes                                         |
|----------------------|-----------------------------------------------|
| Logistic Regression  | Tuned with `GridSearchCV` on ROC AUC         |
| Random Forest        | Tuned with `RandomizedSearchCV`              |
| Support Vector Classifier (SVC) | With `StandardScaler` and hyperparam tuning  |
| Gradient Boosting    | Tuned with `GridSearchCV`                    |

---

## 🧪 Model Evaluation

Each model is evaluated using **Stratified 5-Fold Cross-Validation** on `train.csv` and a final holdout split (80/20) to report:

- Accuracy
- Precision
- Recall
- ROC AUC

Updated evaluation results:

| Model              | Accuracy | Precision | Recall | ROC AUC |
|--------------------|----------|-----------|--------|---------|
| Logistic Regression| 0.888    | 0.773     | 0.638  | 0.927   |
| Random Forest      | 0.889    | 0.812     | 0.598  | 0.929   |
| SVC                | 0.762    | 0.459     | 0.909  | 0.903   |
| Gradient Boosting  | 0.891    | 0.792     | 0.632  | 0.930   |

---

## 📤 Submissions

Each model generates a `submission.csv` with predicted churn probabilities for test customers. Format matches `sample_submission.csv`.

---

## 📦 Dependencies

```bash
pandas
numpy
scikit-learn
matplotlib
seaborn
imblearn
```

Install with:
```bash
pip install -r requirements.txt
```

## 🚀 How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Customer-Churn-Prediction.git
   cd Customer-Churn-Prediction 
   ```
2. Install dependencies
3. Run the notebook or Python scripts to generate submission.csv

## 🤝 Acknowledgments
This project was completed as part of the Lateral AI Academy competition.

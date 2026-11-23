# ML_assignment_2
This is a Machine Learning classification assignment using the Bank Marketing dataset and five classification algorithm

# Telco Customer Churn Prediction Data set


## 1. Problem Statement

Customer churn (customers leaving the service) is costly for telecom providers, since retaining
a customer is cheaper than acquiring a new one. The goal of this project is to build and compare
five classification models that predict whether a customer will **churn (Yes/No)** from their
account, service and billing information. The models are compared on Accuracy, AUC, Precision,
Recall, F1 and Matthews Correlation Coefficient (MCC).

## 2. Dataset Description  [1 mark]

- **Source:** Telco Customer Churn (IBM sample data), Kaggle - https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Why this dataset:** <!-- EDIT: 2-3 lines in your own words, e.g. real business problem, mix of numeric and categorical features, a hidden missing-value issue, and an imbalanced target that makes metric choice interesting -->
- **Size:** 7,043 rows x 21 columns (20 features + target) - meets the 12-feature / 1,000-instance requirement.
- **Target variable:** `Churn` (Yes = customer left, No = stayed) - binary classification.
- **Feature types:**
  - Numeric: `tenure`, `MonthlyCharges`, `TotalCharges`
  - Binary (0/1): `SeniorCitizen`
  - Categorical (15): `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`
  - Identifier (dropped): `customerID`
- **Class distribution:** No = 5,174 (73.46 %), Yes = 1,869 (26.54 %). The target is imbalanced (about 2.77 : 1), so accuracy alone is misleading and AUC, F1 and MCC are also considered.

![Class distribution](outputs/class_distribution.png)


<!-- EDIT: screenshot of the run on BITS Virtual Lab goes in the PDF -->

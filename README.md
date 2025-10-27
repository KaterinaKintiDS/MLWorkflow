# # Machine Learning Workflow – Classification Project

This repository contains a complete end-to-end machine learning workflow for a binary classification problem. The notebook demonstrates how to prepare data, train multiple models, evaluate performance, and interpret results using explainability tools.

---

## 📌 Project Objectives

- Explore and preprocess the dataset  
- Handle duplicates and examine class distribution  
- Train and compare multiple machine learning models  
- Evaluate model performance using standard classification metrics  
- Interpret the best model using **SHAP** values  

---

## 🧠 Models Implemented

| Model | Library | Purpose |
|------|---------|---------|
| Random Forest | `scikit-learn` | Baseline ensemble classifier |
| Logistic Regression | `scikit-learn` | Linear interpretable model |
| XGBoost | `xgboost` / `sklearn` API | Gradient boosting for performance |

---

## 🗂 Workflow Overview

### 1. **Import Dependencies**
The notebook loads key data science libraries such as:
- `pandas`, `numpy`
- `seaborn`, `matplotlib`
- `scikit-learn` classifiers & evaluation metrics
- `XGBoost`
- `SHAP` for explainability

### 2. **Data Exploration**
- Display dataset structure
- Identify missing values
- Check duplicates using:

```python
def calculate_duplicate_percentage(df):
    total_rows = len(df)
    duplicate_rows = len(df[df.duplicated()])
    return (duplicate_rows / total_rows) * 100

MLWorkflow

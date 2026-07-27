# 🚢 Titanic Survival Classification

Predicting passenger survival on the RMS Titanic using a Random Forest Classifier — built as part of the Data Science Internship (Month 1) at **Arch Technologies**.

## 📌 Project Overview

The sinking of the RMS Titanic on April 15, 1912 resulted in the deaths of 1,502 out of 2,224 passengers and crew. This project uses supervised machine learning to predict whether a passenger survived, based on features like age, gender, ticket class, and fare.

## 📊 Dataset

- **Source:** [Titanic dataset — Kaggle](https://www.kaggle.com/c/titanic/data)
- **Records:** 891 passengers, 12 attributes (1 target + 11 features)

## 🧹 Data Cleaning & Feature Engineering

| Column | Missing | Strategy |
|---|---|---|
| Cabin | 687 (77.10%) | Dropped (extreme sparsity) |
| Age | 177 (19.87%) | Imputed with median (28.0 years) |
| Embarked | 2 (0.22%) | Imputed with mode ('S' - Southampton) |

**Engineered features:**
- `FamilySize` = `SibSp` + `Parch` + 1
- `IsAlone` = 1 if `FamilySize` == 1, else 0
- One-hot encoding on `Sex` and `Embarked` (`drop_first=True`)
- `StandardScaler` applied to continuous variables (Age, Fare, FamilySize)

## 🔍 Key EDA Insights

- **Gender:** 74.20% of female passengers survived vs. only 18.89% of male passengers — the single strongest predictor of survival.
- **Ticket Class:** 1st Class → 62.96% survival, 2nd Class → 47.28%, 3rd Class → 24.24%.

## 🤖 Model

- **Algorithm:** Random Forest Classifier (`n_estimators=100`, `max_depth=5`)
- **Split:** 80% train (712 records) / 20% stratified test (179 records)

## 📈 Results

| Metric | Score |
|---|---|
| Accuracy | **79.33%** |
| Precision | **79.63%** |
| F1-Score | **69.92%** |

**Confusion Matrix:** Of 110 actual deceased passengers, 99 were correctly identified; of 69 actual survivors, 43 were correctly predicted.

**Top Feature Importances:**
1. Sex_male — 42.07%
2. Fare — 16.95%
3. Pclass — 13.69%
4. Age — 11.12%

## 🖼️ Visualizations

| Gender Survival | Class Survival |
|---|---|
| ![Gender](images/chart_1_gender_survival.png) | ![Pclass](images/chart_2_pclass_survival.png) |

| Confusion Matrix | Feature Importance |
|---|---|
| ![Confusion Matrix](images/chart_3_confusion_matrix.png) | ![Feature Importance](images/chart_4_feature_importance.png) |

## 🛠️ Tech Stack

`Python` · `pandas` · `numpy` · `scikit-learn` · `matplotlib` · `seaborn`

## 🚀 How to Run

```bash
git clone https://github.com/your-username/titanic-survival-classification.git
cd titanic-survival-classification
pip install -r requirements.txt
python Titanic_Survival_Classification.py
```

## 📄 Full Report

The detailed project report (with methodology, screenshots, and code walkthrough) is available in [`report/`](report/).

## 🔮 Future Enhancements

- Extract passenger **Titles** (Mr, Miss, Master, Dr) from the Name field for added signal
- Hyperparameter tuning via `GridSearchCV`
- Test alternative models such as **XGBoost** or **LightGBM** to improve recall on the positive (Survived) class

---
**Author:** Fatima Riaz · Data Science Internship, Arch Technologies

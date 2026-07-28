
# ===========================================================
#                 Titanic Survival Classification
# ===========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Display settings so all table columns print completely in terminal
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# -----------------------------------------------------------
# 1. LOAD DATA & INITIAL SUMMARY
# -----------------------------------------------------------
df = pd.read_csv(r'c:\Users\NEW LAPTOP CITY\.vscode\Internship Projects\2. Project 1\train.csv')

print("="*70)
print(" 1. INITIAL DATASET OVERVIEW (FOR INTRODUCTION SECTION)")
print("="*70)
print(f"Total Rows (Passengers): {df.shape[0]}")
print(f"Total Columns (Features): {df.shape[1]}\n")

print("--- Sample Data (First 5 Rows) ---")
print(df.head())

print("\n--- Data Structure & Non-Null Counts ---")
print(df.info())

print("\n--- Numerical Summary Statistics ---")
print(df.describe())

# -----------------------------------------------------------
# 2. MISSING VALUE ANALYSIS
# -----------------------------------------------------------
print("\n" + "="*70)
print(" 2. MISSING VALUE ANALYSIS (FOR DATA CLEANING SECTION)")
print("="*70)
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'Missing Values': missing, 'Percentage (%)': missing_pct})
print(missing_df[missing_df['Missing Values'] > 0].sort_values(by='Missing Values', ascending=False))

# -----------------------------------------------------------
# 3. COMPREHENSIVE EXPLORATORY DATA ANALYSIS (EDA)
# -----------------------------------------------------------
print("\n" + "="*70)
print(" 3. DETAILED EXPLORATORY STATISTICS (FOR REPORT BODY)")
print("="*70)

# Overall Target Distribution
overall_surv = df['Survived'].value_counts()
overall_surv_pct = df['Survived'].value_counts(normalize=True) * 100
print("--- Overall Target Distribution (Survived) ---")
print(f"Died (0)    : {overall_surv[0]} passengers ({overall_surv_pct[0]:.2f}%)")
print(f"Survived (1): {overall_surv[1]} passengers ({overall_surv_pct[1]:.2f}%)")

# Survival by Gender
print("\n--- Survival Rate by Gender ---")
gender_ct = pd.crosstab(df['Sex'], df['Survived'], margins=True, margins_name='Total')
gender_ct['Survival Rate (%)'] = (pd.crosstab(df['Sex'], df['Survived'], normalize='index')[1] * 100)
print(gender_ct)

# Survival by Passenger Class
print("\n--- Survival Rate by Ticket Class (Pclass) ---")
pclass_ct = pd.crosstab(df['Pclass'], df['Survived'], margins=True, margins_name='Total')
pclass_ct['Survival Rate (%)'] = (pd.crosstab(df['Pclass'], df['Survived'], normalize='index')[1] * 100)
print(pclass_ct)

# Survival by Embarkation Port
print("\n--- Survival Rate by Embarkation Port ---")
embarked_ct = pd.crosstab(df['Embarked'], df['Survived'], margins=True, margins_name='Total')
embarked_ct['Survival Rate (%)'] = (pd.crosstab(df['Embarked'], df['Survived'], normalize='index')[1] * 100)
print(embarked_ct)

# Age Statistics by Survival
print("\n--- Detailed Age Statistics by Survival ---")
print(df.groupby('Survived')['Age'].describe())

# Fare Statistics by Survival
print("\n--- Detailed Fare Statistics by Survival ---")
print(df.groupby('Survived')['Fare'].describe())

# -----------------------------------------------------------
# 4. DATA PREPROCESSING & FEATURE ENGINEERING
# -----------------------------------------------------------
data = df.copy()

# Impute missing values
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

# Drop high-missing or non-predictive columns
data.drop(columns=['Cabin', 'PassengerId', 'Name', 'Ticket'], inplace=True)

# Feature Engineering
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
data['IsAlone'] = (data['FamilySize'] == 1).astype(int)

# One-hot encoding
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

# -----------------------------------------------------------
# 5. MODEL TRAINING & EVALUATION
# -----------------------------------------------------------
X = data.drop(columns=['Survived'])
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("\n" + "="*70)
print(" 4. MODEL EVALUATION METRICS (FOR RESULTS SECTION)")
print("="*70)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy : {acc * 100:.2f}%")
print(f"Precision: {prec * 100:.2f}%")
print(f"Recall   : {rec * 100:.2f}%")
print(f"F1-Score : {f1 * 100:.2f}%")

print("\n--- Full Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['Died', 'Survived']))

print("--- Confusion Matrix Breakdown ---")
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=['Actual Died', 'Actual Survived'], columns=['Pred Died', 'Pred Survived'])
print(cm_df)

# Feature Importance Breakdown
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n--- Feature Importance Breakdown ---")
print((importances * 100).map('{:.2f}%'.format))

# -----------------------------------------------------------
# 6. GENERATE AND SAVE SEPARATE GRAPH IMAGES
# -----------------------------------------------------------
cm_plot = confusion_matrix(y_test, y_pred)
importances_plot = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)

# Graph 1: Gender Survival
plt.figure(figsize=(7, 5))
sns.barplot(x='Sex', y='Survived', data=df, hue='Sex', palette='Set2', legend=False)
plt.title('Survival Rate by Gender', fontsize=14, fontweight='bold')
plt.xlabel('Gender', fontsize=12)
plt.ylabel('Survival Probability', fontsize=12)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('chart_1_gender_survival.png', dpi=300)
plt.close()

# Graph 2: Pclass Survival
plt.figure(figsize=(7, 5))
sns.barplot(x='Pclass', y='Survived', data=df, hue='Pclass', palette='Set1', legend=False)
plt.title('Survival Rate by Passenger Class (Pclass)', fontsize=14, fontweight='bold')
plt.xlabel('Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)', fontsize=12)
plt.ylabel('Survival Probability', fontsize=12)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('chart_2_pclass_survival.png', dpi=300)
plt.close()

# Graph 3: Confusion Matrix
plt.figure(figsize=(7, 5))
sns.heatmap(cm_plot, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Died', 'Survived'], 
            yticklabels=['Died', 'Survived'],
            annot_kws={"size": 14})
plt.title('Random Forest - Confusion Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('Actual Label', fontsize=12)
plt.tight_layout()
plt.savefig('chart_3_confusion_matrix.png', dpi=300)
plt.close()

# Graph 4: Feature Importance
plt.figure(figsize=(8, 6))
importances_plot.plot(kind='barh', color='teal')
plt.title('Random Forest - Feature Importance', fontsize=14, fontweight='bold')
plt.xlabel('Relative Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.savefig('chart_4_feature_importance.png', dpi=300)
plt.close()

print("\n[SUCCESS] All prints complete and graphs saved individually!")

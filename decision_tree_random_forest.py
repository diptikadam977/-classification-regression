# decision_tree_random_forest.py
# Decision Tree and Random Forest pipeline on Breast Cancer dataset.

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

def main():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    dt = DecisionTreeClassifier(random_state=42, max_depth=4)
    dt.fit(X_train_s, y_train)
    y_pred_dt = dt.predict(X_test_s)
    acc_dt = accuracy_score(y_test, y_pred_dt)
    print('Decision Tree Test Accuracy:', acc_dt)
    print('Decision Tree Classification Report:\n', classification_report(y_test, y_pred_dt, target_names=data.target_names))

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_s, y_train)
    y_pred_rf = rf.predict(X_test_s)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print('Random Forest Test Accuracy:', acc_rf)
    print('Random Forest Classification Report:\n', classification_report(y_test, y_pred_rf, target_names=data.target_names))

if __name__ == '__main__':
    main()

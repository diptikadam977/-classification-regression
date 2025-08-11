
# Decision Trees & Random Forests — Internship Task

## Overview
This project demonstrates Decision Tree and Random Forest classifiers using the Breast Cancer Wisconsin dataset.

## Files
- `decision_tree_random_forest.py` — runnable script for training/evaluating models.
- `decision_tree.png` — visualization of the trained Decision Tree.
- `feature_importances.png` — top 15 feature importances from the Random Forest.
- `model_results.csv` — accuracy and cross-validation summary.
- `Decision_Tree_Task.pdf` — final report (included below).

## How to run
```
pip install numpy pandas scikit-learn matplotlib reportlab
python decision_tree_random_forest.py
```

## Quick Results
Decision Tree test accuracy: 0.9386
Random Forest test accuracy: 0.9561
Decision Tree CV mean accuracy: 0.9227 ± 0.0295
Random Forest CV mean accuracy: 0.9561 ± 0.0123

## Interview Q&A (short)
1. Decision tree works by recursively splitting features to reduce impurity (entropy/gini) until stopping criteria.
2. Entropy measures disorder; information gain is the reduction in entropy after a split.
3. Random forest reduces variance by averaging many trees trained on bootstrapped samples (bagging) and feature subsampling.
4. Overfitting: model learns noise; prevent by pruning, setting max_depth, min_samples_leaf, or using ensembles.
5. Bagging: bootstrap aggregating — training models on different random subsets and averaging.
6. Visualize using `plot_tree` or export_graphviz + Graphviz.
7. Feature importance: higher value means feature contributed more to splitting decisions.
8. Pros/Cons: Robust and accurate but less interpretable and heavier compute.


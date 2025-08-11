# **README — Decision Trees & Random Forests Project**

## 1. Overview

This project compares two popular machine learning algorithms — **Decision Tree Classifier** and **Random Forest Classifier**.
We train both models on a dataset, evaluate their performance, and identify which performs better for classification.

---

## 2. Algorithms Used

* **Decision Tree**: A model that splits data into branches based on feature values to make predictions.
* **Random Forest**: An ensemble method that builds multiple Decision Trees and averages their predictions to improve accuracy and reduce overfitting.

---

## 3. Project Workflow

1. **Import Libraries** — pandas, numpy, matplotlib, seaborn, scikit-learn
2. **Load Dataset** — Read the CSV file into a DataFrame
3. **Exploratory Data Analysis (EDA)** — Understand the dataset with summaries, visualizations, and missing value checks
4. **Data Preprocessing** — Encode categorical features, handle missing values
5. **Split Data** — Train/test split
6. **Model Building**

   * Train a Decision Tree Classifier
   * Train a Random Forest Classifier
7. **Evaluation** — Use accuracy score, confusion matrix, and classification report
8. **Comparison** — Analyze which model performs better
9. **Conclusion**

---

## 4. Files Included

* `Decision_Tree_Random_Forest.ipynb` — Jupyter Notebook with complete code and results
* `dataset.csv` — Dataset used in the project
* `README.txt` — Project documentation
* `Project_Report.pdf` — Exported report (optional)

---

## 5. Requirements

Install dependencies before running:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## 6. How to Run

1. Place `dataset.csv` in the same directory as the script.
2. Open `Decision_Tree_Random_Forest.py` in  VS Code.
3. Run the cells in order.
4. View accuracy scores, confusion matrices, and classification reports.

---

## 7. Output

* Accuracy score for each model
* Confusion matrix plots
* Classification reports (precision, recall, F1-score)
* Comparison table summarizing results

---

## 8. Conclusion

Random Forest generally outperforms a single Decision Tree due to its ability to reduce overfitting and improve prediction accuracy.
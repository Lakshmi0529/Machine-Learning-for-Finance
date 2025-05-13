Goal: Implementing a K-Nearest Neighbors (KNN) classifier on a finance-related dataset: the Bank Marketing Dataset from the UCI Machine Learning Repository and predicting whether a customer will subscribe to a term deposit (yes: 1, No: 0). 

Dataset Overview:
This dataset contains information about clients contacted through marketing campaigns and whether they subscribed to a term deposit.
Features: Includes attributes like age, job, marital status, education, default, balance, housing loan, personal loan, contact communication type, last contact day and month, duration, campaign, pdays, previous, poutcome, and more.

Target Variable: y — indicates if the client subscribed to a term deposit (yes or no).
To figure out the best value for 'k'(n_neighbors) I used GridSearchCV by running 5-fold cross-validation.
best parameters are: n_neighbors(k) = 9, distance metric: 'euclidean', and weights: 'distance'

Comparison: Before vs After Tuning
Metric	Class 1 (Before)	Class 1 (After)
Precision	0.63	0.65
Recall	0.45	0.42
F1-Score	0.52	0.51

Accuracy stayed the same (0.91)

Class 0 metrics stayed high (model is good at predicting "no")

Class 1 (minority class) saw a tiny precision gain, but recall slightly dropped, so F1 didn’t improve

Interpretation

KNN with hyperparameter tuning didn't significantly improve recall or F1-score for class 1.

This is expected in imbalanced datasets, especially with models like KNN, which:

Are sensitive to class imbalance

Can be biased toward the majority class due to vote-based nature

To correct for class imbalance I made use of SMOTE to improve recall for class 1 as can be seen in the classification report

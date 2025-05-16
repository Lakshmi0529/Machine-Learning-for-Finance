Dataset: IEEE-CIS Fraud Detection (Kaggle)
Goal: Predict fraudulent transactions using transaction and identity data.

The data is divided into 2 csv files Transaction and identity, which are joined by transactionID column. The objective
is to predict whether a given transaction isFraud(class 1) and isNotFraud(class0). I used RandomForestClassifier from sklearn
library to classify the target variable.

However, the RandomForestClassifier does not correct for Class imbalance leading to low precision on class 1 but a higher recall.
To fix this I used XGBoost to correct for class imbalance.

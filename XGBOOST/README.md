This project uses the IEEE-CIS Fraud Detection dataset from Kaggle.

The objective is to detect whether a transaction is fraudulent (Class 1) or not (Class 0).
The data has 219 features excluding the target variable and the transactionID column which is used to join the transaction and identity datasets.
The XGBClassifier corrects the problem of class imbalance by scaling the weights of class 1 and class 0 samples.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix,classification_report,roc_auc_score,roc_curve
import xgboost as xgb

# Loading the dataset
train_transaction = pd.read_csv(r"D:\ML\Random Forests\train_transaction.csv")
train_identity = pd.read_csv(r"D:\ML\Random Forests\train_identity.csv")

train = pd.merge(train_transaction,train_identity,on='TransactionID',how="left")
mean_fraud_rate = train['isFraud'].mean()
print(f'{mean_fraud_rate:.2%}')

# Data Preprocessing
    # Handling missing values
train = train.loc[:,train.isnull().mean() < 0.5] # Drops columns with more than 50% missing values
    # filling missing values
for col in train.select_dtypes(include="Float64").columns:
    train[col].fillna(train[col].median(),inplace=True)

for col in train.select_dtypes(include="object").columns:
    train[col].fillna('Unknown',inplace=True)

# Feature Engineering
train['TransactionHour'] = train['TransactionDT'] // 3600 % 24

for col in train.select_dtypes(include='object').columns:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))

# Train-Test Split
X = train.drop(['isFraud','TransactionID'],axis=1)
y = train['isFraud']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

# Building the Model
scale_pos_weight = len(y_train[y_train == 0])/len(y_train[y_train == 1])

model = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    n_estimators=300,
    max_depth = 6,
    learning_rate =0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='aucpr',  # Optimize for precision-recall (better for fraud)
    random_state=42,
    tree_method = 'hist'
    )

model.fit(X_train,y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]
roc_auc = roc_auc_score(y_test,y_proba)

#  Evaluation
print(classification_report(y_test,y_pred))
print(f'ROC-AUC: {roc_auc:.4f}')

disp = ConfusionMatrixDisplay(confusion_matrix(y_test,y_pred),display_labels=['isFraud','isNotFraud'])
disp.plot()
plt.show()

fpr,tpr,thresholds = roc_curve(y_test,y_proba)

plt.figure()
plt.plot(fpr,tpr,label=f'ROC Curve (area = {roc_auc:.2f})')
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristics')
plt.legend(loc='lower right')
plt.show()
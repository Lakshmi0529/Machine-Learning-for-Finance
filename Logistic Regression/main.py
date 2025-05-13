# Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import classification_report,ConfusionMatrixDisplay,confusion_matrix,roc_curve,roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Data loading and exploration

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
column_names = [
    'checking_account', 'duration', 'credit_history', 'purpose', 'credit_amount',
    'savings_account', 'employment', 'installment_rate', 'personal_status',
    'other_debtors', 'residence_since', 'property', 'age', 'other_installment_plans',
    'housing', 'existing_credits', 'job', 'people_liable', 'telephone', 'foreign_worker',
    'credit_risk'
]

data = pd.read_csv(url, delimiter=' ', header=None, names=column_names)
# Converting target to binary (1: Good, 0: Bad)
data['credit_risk'] = data['credit_risk'].map({1: 0,2: 1})

# print(data.head())
# print(data.info())
# print(data.describe())
# print(data['credit_risk'].value_counts(normalize=True))

# Split the data
X = data.drop('credit_risk',axis=1)
y = data['credit_risk']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)

# Data Preprocessing

categorical_cols = X_train.select_dtypes(include=['object']).columns.to_list()
numerical_cols = X_train.select_dtypes(include=['int64']).columns.to_list()

# Create preprocessing pipelines

numerical_transformer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='median')),
    ('scaler',StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('onehot',OneHotEncoder(handle_unknown='ignore'))])

# Combine preprocessing steps

preprocessor = ColumnTransformer(transformers=[
    ('num',numerical_transformer,numerical_cols),
    ('cat',categorical_transformer,categorical_cols)])


# Model Building
model = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('classifier',LogisticRegression(solver='liblinear',random_state=42))])

model.fit(X_train,y_train)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:,1]

# Model Evaluation

print('Classification Report: ')
print(classification_report(y_test,y_pred=y_pred))

cm = confusion_matrix(y_test,y_pred)
disp = ConfusionMatrixDisplay(cm,display_labels=['Good Credit','Bad Credit'])
disp.plot()
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_true=y_test,y_score=y_pred_proba)
roc_auc = roc_auc_score(y_true=y_test,y_score=y_pred_proba)

plt.figure()
plt.plot(fpr,tpr,label=f'ROC Curve (area = {roc_auc:.2f})')
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('Receiver Operating Characteristics')
plt.legend(loc='lower right')
plt.show()

# Feature Importance

preprocessor.fit(X_train)
numeric_features = numerical_cols
categorical_features = list(
    model.named_steps['preprocessor']
    .named_transformers_['cat']
    .named_steps['onehot']
    .get_feature_names_out(categorical_cols)
)  # Get feature names after one hot encoding

all_features = numeric_features + categorical_features

# Get coefficients from the Trained model
coefficients = model.named_steps['classifier'].coef_[0]

# create a dataframe for feature importance
feature_importance = pd.DataFrame({
    'Feature': all_features,
    'Importance': coefficients
}).sort_values('Importance',key=abs,ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance.head(20),
    palette='coolwarm'
)
plt.title('Top 20 Most Important Features (Absolute Coefficient Values)')
plt.xlabel('Coefficient Magnitude')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
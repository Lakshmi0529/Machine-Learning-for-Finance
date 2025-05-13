# Importing Libraries

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay,roc_auc_score,roc_curve
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from scipy.stats import ttest_rel
# Loading data
path = r"D:\ML\KNN Classifier\bank-additional\bank-additional-full.csv"
data = pd.read_csv(path, sep=';')

# Split the data
X = data.drop('y',axis=1)
y = data['y'].map({'no': 0,'yes': 1})
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

# Data preprocessing
categorical_cols = X_train.select_dtypes(include=['object']).columns.to_list()
numerical_cols = X_train.select_dtypes(exclude=['object']).columns.to_list()

# Create Preprocessing Pipelines
numerical_transformer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='median')),
    ('scaler',StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('onehot',OneHotEncoder(handle_unknown='ignore'))])

# Combining Preprocessing steps
preprocessor = ColumnTransformer(transformers=[
    ('num',numerical_transformer,numerical_cols),
    ('cat',categorical_transformer,categorical_cols)])

# Building the model
model = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('smote',SMOTE(random_state=42))
    ('classifier',KNeighborsClassifier(n_neighbors=9,weights='distance',metric='euclidean'))])
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

# Model Evaluation
cm = confusion_matrix(y_test,y_pred)
disp = ConfusionMatrixDisplay(cm,display_labels=['Yes','No'])
disp.plot()
plt.title('Confusion Matrix')
plt.show()
print('Classification Report:\n',classification_report(y_test,y_pred))

# Plotting ROC-AUC Curve
y_pred_proba = model.predict_proba(X_test)[:,1]
fpr,tpr,thresholds = roc_curve(y_true=y_test,y_score=y_pred_proba)
roc_auc = roc_auc_score(y_true=y_test,y_score=y_pred_proba)

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

# Hyperparameter Tuning for 'k'
param_grid = {
    'classifier__n_neighbors': list(range(3,21,2)),
    'classifier__weights': ['uniform','distance'],
    'classifier__metric': ['euclidean','manhattan']
}

grid_search = GridSearchCV(model,param_grid=param_grid,cv=5,scoring='f1',n_jobs=-1)
grid_search.fit(X_train,y_train)
print(grid_search.best_params_)

# Using SMOTE to correct for class imbalance
model_a = ImbPipeline(steps=[
    ('preprocessor',preprocessor),
    ('smote',SMOTE(random_state=42)),
    ('classifier',KNeighborsClassifier(n_neighbors=9,metric='euclidean',weights='distance'))])

f1_score_model = cross_val_score(model,X_train,y_train,scoring='f1',cv=5)
f1_score_model_a = cross_val_score(model_a,X_train,y_train,scoring='f1',cv=5)

print(f"f1-score without smote: {f1_score_model}")
print(f"f1-score with smote: {f1_score_model_a}")

t_stat,p_value = ttest_rel(f1_score_model_a,f1_score_model)


if p_value < 0.05:
    print("✅ Statistically significant: SMOTE improved the model.")
else:
    print("⚠️ No statistically significant difference.")
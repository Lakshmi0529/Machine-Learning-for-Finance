# Importing the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r"D:\ML\archive (1)\Ecommerce Customers.csv")

# Inspecting the data

print(df.head())
print(df.isnull().sum())

print(df.describe())

sns.pairplot(df)
plt.show()
print(df.columns.to_list())

# defining feature and target variables
X = df[['Time on App','Length of Membership']]
y = df['Yearly Amount Spent']

X_sm = sm.add_constant(X)
model = sm.OLS(y,X_sm).fit()
results = print(model.summary())

# Using sklearn

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)

predictions = model.predict(X_test)
coef_df = pd.DataFrame({'Feature': X.columns.to_list(),
                         'Coefficient':model.coef_})
print(coef_df)
print(f'R-squared: {round(r2_score(y_test,predictions),3)}')

# Visualizing the model

plt.scatter(y_test, predictions, alpha=0.5)
plt.xlabel("Actual Yearly Amount Spent")
plt.ylabel("Predicted Yearly Amount Spent")
plt.title("Actual vs. Predicted Values")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.show()

# Check for Violation of assumptions

# Linearity (No pattern should exist in the plot Fittedvalues Vs Residuals)

residuals = model.resid
fitted = model.fittedvalues

plt.figure(figsize=(8,5))
sns.scatterplot(x=fitted,y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs Fitted')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.show()

# # Homoscedasticity

plt.figure(figsize=(8, 5))
sns.scatterplot(x=fitted, y=np.sqrt(np.abs(residuals)))
plt.title('Scale-Location')
plt.xlabel('Fitted values')
plt.ylabel('√|Standardized residuals|')
plt.show()

# # QQ Plot (Check for normality of residuals)

stats.probplot(residuals,dist='norm',plot=plt)
plt.title('Normal Q-Q')
plt.show()

# check for multicollinearity

vif_data = pd.DataFrame()
vif_data['feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
print(vif_data)

# Running PCA to reduce high multicollinearity
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(pca.explained_variance_ratio_)
print("Cumulative Variance:", sum(pca.explained_variance_ratio_))

# Predicting amount spent

df['Predicted Spend'] = model.predict(X_sm)
df['Residual'] = df['Yearly Amount Spent'] - df['Predicted Spend']

df.to_csv('regression.csv',index=False)

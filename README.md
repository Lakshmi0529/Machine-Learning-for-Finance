This project demonstrates how linear regression can be applied to real-world financial data for predictive analysis, a common task in FP&A.

For this project I used the E-commerce Customers dataset, which contains information about customers' interactions with an e-commerce platform, including:

Avg. Session Length: Average session length in minutes

Time on App: Average time spent on the app in minutes

Time on Website: Average time spent on the website in minutes

Length of Membership: Number of years the customer has been a member

Yearly Amount Spent: Amount of money spent by the customer in a year (target variable)

Both Statsmodels and Sklearn were used to predict the target variable (Yearly Amount Spent)

I also implemented checks for violation of assumptions of Linear Regression.
PCA was used to reduce high multicollinearity as evidenced by the high variance inflation factor calculated.

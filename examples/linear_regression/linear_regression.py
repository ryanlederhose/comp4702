import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("linear_regression.csv")

# Drop NaN
df = df.dropna()

# Get input features
input_cols = df.columns.tolist()
input_cols.pop(-1)
print("Input columns: ", input_cols)

# Normalise input features
scaler = StandardScaler()
scaler.fit(df[input_cols])
df[input_cols] = scaler.transform(df[input_cols])

# Split the data into features and target variable
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print(X)
print(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Build Linear Regression model
linear_regression = LinearRegression()
model = linear_regression.fit(X_train, y_train)

# Print model coefficients
print("Model Coefficients: ", model.coef_)
print("Model Intercept: ", model.intercept_)

# Predict test set
y_pred = model.predict(X_test)

# Calculate SSE
sse = np.sum((y_pred - y_test)**2)
print("SSE: ", sse)

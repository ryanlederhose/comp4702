import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


FILENAME = "regression_tree.csv"

# Load the CSV file into a pandas DataFrame
data = pd.read_csv(FILENAME)

# Shuffle the rows randomly
data = data.sample(frac=1, random_state=42)

# Split the dataset into features and target variable
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Combine the test data
combined_test = pd.concat([X_test, y_test], axis=1)
combined_test = combined_test.sort_values(by='x', ascending=True)

# Create the Decision Tree Regressor
depth = 11
clf = DecisionTreeRegressor(max_depth=depth)

# Fit the model to the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test.sort_values(by='x', ascending=True))

# Compute the test SSE
test_sse = np.sum((combined_test['y'] -
                  y_pred)**2)
print('Test loss (SSE):', test_sse)

# Plot the training and test data together with the predicted function of the model
fig, ax = plt.subplots()
ax.scatter(X_train, y_train, label='Training data')
ax.scatter(X_test,
           y_test, label='Test data')
ax.plot(combined_test['x'],
        y_pred, label='Predicted function')
ax.legend()
plt.show()

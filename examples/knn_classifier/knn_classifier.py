import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import zero_one_loss
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import confusion_matrix

FILENAME = "knn_classifier.csv"

# Load the CSV file into a pandas DataFrame
data = pd.read_csv(FILENAME)

# Split the data into features and target variable
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Create a k-NN classifier with k=3
k = 1
knn = KNeighborsClassifier(n_neighbors=k)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Make predictions on the test data
y_test_pred = knn.predict(X_test)

# Compute the confusion matrix to get the misclassification rate
cm = confusion_matrix(y_test, y_test_pred)
misclassification_rate = (cm[0, 1] + cm[1, 0]) / len(y_test)

print('Misclassification rate:', misclassification_rate)

# Plot the decision regions for training data
plot_decision_regions(X_train.to_numpy(),
                      y_train.to_numpy(), clf=knn, legend=2)
plt.title('Training Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Plot the decision regions for test data
plot_decision_regions(X_test.to_numpy(), y_test.to_numpy(), clf=knn, legend=2)
plt.title('Test Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

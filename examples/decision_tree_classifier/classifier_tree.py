import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

FILENAME = "classifier_tree.csv"

# Load the CSV file into a pandas DataFrame
data = pd.read_csv(FILENAME)

# Split the data into features and target variable
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Create the decision tree classifier
clf = DecisionTreeClassifier(max_depth=3)

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Calculate the misclassification rate
misclassification_rate = 1 - accuracy_score(y_test, y_pred)

# Print the misclassification rate
print('Misclassification rate:', misclassification_rate)

# Plot the decision regions for training data
plot_decision_regions(X_train.to_numpy(),
                      y_train.to_numpy(), clf=clf, legend=2)
plt.title('Training Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Plot the decision regions for test data
plot_decision_regions(X_test.to_numpy(), y_test.to_numpy(),
                      clf=clf, legend=2)
plt.title('Test Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

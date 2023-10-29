import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
import statistics as stat

FILENAME = "cross_val.csv"

# Load the CSV file into a pandas DataFrame
data = pd.read_csv(FILENAME, names=['feature1', 'feature2', 'class'])
print(data.columns)

# Shuffle the rows randomly
data = data.sample(frac=1, random_state=42)

# Split the data\
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Print the data
print(X)
print(y)

# Create knn Model
knn = KNeighborsClassifier(n_neighbors=20)

# Initialise the 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Perform 10-fold cross validation
score = cross_val_score(knn, X, y, cv=kf)

# Print average accuracy
print("Average Accuracy: ", score.mean())
print("Standard Deviation: ", np.std(score))
'''
Average accuracy is 0.9574999
Standard deviation is 0.033634060117684265
'''

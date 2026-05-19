# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import dataset
from sklearn.datasets import load_iris

# Split dataset
from sklearn.model_selection import train_test_split

# KNN Algorithm
from sklearn.neighbors import KNeighborsClassifier

# Accuracy checker
from sklearn.metrics import accuracy_score

# Load iris dataset
iris = load_iris()

# Features and target
X = iris.data
y = iris.target

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Create model
model = KNeighborsClassifier(n_neighbors=3)

# Train model
model.fit(X_train, y_train)

# Predict test data
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

# Predict custom flower
# sample = [[5.1, 3.5, 1.4, 0.2]]
sample = [[6.5, 3.0, 5.2, 2.0]]

prediction = model.predict(sample)

print("Predicted Flower:", iris.target_names[prediction][0])

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Iris Dataset")
plt.show()
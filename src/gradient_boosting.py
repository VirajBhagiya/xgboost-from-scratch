# src/gradient_boosting.py

import numpy as np
from .decision_tree import DecisionTree

class GradientBoosting:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2):
        """
        Initialize gradient boosting parameters.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        """
        Train the model by iteratively adding trees.
        """
        # Initialize predictions with the mean of the target values
        initial_prediction = np.mean(y)
        predictions = np.full(y.shape, initial_prediction)

        for _ in range(self.n_estimators):
            # Compute residuals (negative gradients)
            residuals = y - predictions

            # Train a tree on the residuals
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X, residuals)

            # Predict using the new tree and scale by learning rate
            update = tree.predict(X)
            predictions += self.learning_rate * update
            self.trees.append(tree)

    def predict(self, X):
        """
        Make predictions by summing contributions from all trees.
        """
        # Start with the initial prediction (average of y)
        predictions = np.zeros(X.shape[0]) + self.initial_prediction

        # Add contributions from each tree
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)

        return predictions

import numpy as np
from .decision_tree import DecisionTree
from .utils import calculate_rmse

class GradientBoosting:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, early_stopping_rounds=5, min_samples_split=2, reg_lambda=1.0, reg_alpha=0.0, colsample=1.0, verbose=False, **kwargs):
        
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.colsample = colsample
        self.verbose = verbose
        self.early_stopping_rounds = early_stopping_rounds
        self.best_iteration = 0
        self.trees = []
        self.initial_prediction = None

    def fit(self, X, y):
        best_score = float("inf")
        no_improve_count = 0
        # Initialize predictions with the mean of the target values
        self.initial_prediction = np.mean(y)
        predictions = np.full(y.shape, self.initial_prediction)
        
        for i in range(self.n_estimators):
            if self.verbose:
                print(f"Fitting estimator {i + 1}/{self.n_estimators}")

            # Compute residuals (negative gradients)
            residuals = y - predictions

            # Randomly sample columns if colsample < 1.0
            n_features = X.shape[1]
            feature_indices = np.random.choice(n_features, int(n_features * self.colsample), replace=False)
            X_subsample = X[:, feature_indices]

            # Train a tree on the residuals
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_subsample, residuals)

            # Predict using the new tree and scale by learning rate
            update = tree.predict(X_subsample)
            predictions += self.learning_rate * update

            # Store the tree and its sampled features
            self.trees.append((tree, feature_indices))

            current_score = calculate_rmse(y, predictions)
            if current_score < best_score:
                best_score = current_score
                self.best_iteration = i
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            # Check for early stopping
            if no_improve_count >= self.early_stopping_rounds:
                print(f"Early stopping at iteration {i}")
                break
            
    def predict(self, X):
        # Start predictions with the initial value
        predictions = np.full(X.shape[0], self.initial_prediction)

        # Add contributions from each tree
        for tree, feature_indices in self.trees:
            predictions += self.learning_rate * tree.predict(X[:, feature_indices])
        
        return predictions
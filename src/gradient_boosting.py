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
        self.best_iteration = n_estimators
        self.trees = []
        self.initial_prediction = None

    def fit(self, X, y):
        best_score = float("inf")
        no_improve_count = 0
        self.initial_prediction = np.mean(y)
        predictions = np.full(y.shape, self.initial_prediction)
        
        for i in range(self.n_estimators):
            if self.verbose:
                print(f"Fitting estimator {i + 1}/{self.n_estimators}")

            residuals = y - predictions
            n_features = X.shape[1]
            feature_indices = np.random.choice(n_features, int(n_features * self.colsample), replace=False)
            X_subsample = X[:, feature_indices]

            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_subsample, residuals)
            update = self.learning_rate * tree.predict(X_subsample)
            predictions += update
            self.trees.append((tree, feature_indices))

            current_score = calculate_rmse(y, predictions)
            if current_score < best_score:
                best_score = current_score
                self.best_iteration = i + 1
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            if no_improve_count >= self.early_stopping_rounds:
                print(f"Early stopping at iteration {i + 1} with RMSE: {best_score:.4f}")
                break
            
    def predict(self, X):
        predictions = np.full(X.shape[0], self.initial_prediction)

        for i, (tree, feature_indices) in enumerate(self.trees[:self.best_iteration]):
            predictions += self.learning_rate * tree.predict(X[:, feature_indices])
        
        return predictions
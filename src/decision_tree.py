import numpy as np

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self._predict_single_input(x) for x in X])

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        
        # Terminate if max depth reached, or samples are insufficient for a split
        if (depth >= self.max_depth or n_samples < self.min_samples_split or len(np.unique(y)) == 1):
            return {"leaf": True, "value": self._calculate_leaf_value(y)}
        
        # Find the best split
        split = self._split(X, y)
        if split is None:
            return {"leaf": True, "value": self._calculate_leaf_value(y)}

        # Recursively build the left and right subtrees
        left_tree = self._build_tree(X[split["left_mask"]], y[split["left_mask"]], depth + 1)
        right_tree = self._build_tree(X[split["right_mask"]], y[split["right_mask"]], depth + 1)

        return {
            "leaf": False,
            "feature_index": split["feature_index"],
            "threshold": split["threshold"],
            "left": left_tree,
            "right": right_tree
        }

    def _split(self, X, y):
        best_mse = float("inf")
        best_split = None
        n_samples, n_features = X.shape

        for feature_index in range(n_features):
        # Sort by feature and get midpoints as candidate thresholds
            sorted_indices = X[:, feature_index].argsort()
            X_sorted, y_sorted = X[sorted_indices, feature_index], y[sorted_indices]
            
            for i in range(1, len(X_sorted)):
                # Check midpoints between adjacent sorted values
                if X_sorted[i] == X_sorted[i - 1]:
                    continue
                threshold = (X_sorted[i] + X_sorted[i - 1]) / 2

                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue

                y_left, y_right = y[left_mask], y[right_mask]
                mse = self._calculate_mse(y_left, y_right)

                if mse < best_mse:
                    best_mse = mse
                    best_split = {
                        "feature_index": feature_index,
                        "threshold": threshold,
                        "left_mask": left_mask,
                        "right_mask": right_mask
                    }

        return best_split

    def _calculate_mse(self, y_left, y_right):
        mse_left = np.var(y_left) * len(y_left)
        mse_right = np.var(y_right) * len(y_right)
        total_mse = (mse_left + mse_right) / (len(y_left) + len(y_right))
        return total_mse

    def _calculate_leaf_value(self, y):
        return np.mean(y)

    def _predict_single_input(self, x):
        node = self.tree
        while not node["leaf"]:
            if x[node["feature_index"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["value"]
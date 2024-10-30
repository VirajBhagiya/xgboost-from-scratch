import numpy as np

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        """
        Initialize the decision tree parameters.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def fit(self, X, y):
        """
        Build the decision tree by recursively splitting the data.
        """
        self.tree = self._build_tree(X, y)
    
    def _build_tree(self, X, y, depth=0):
        """
        Recursive function to build the tree structure.
        """
        # Placeholder for recursive tree-building logic
        pass

    def predict(self, X):
        """
        Predict values for the input data X.
        """
        return np.array([self._predict_single_input(x) for x in X])

    def _predict_single_input(self, x):
        """
        Traverse the tree to predict a single data point.
        """
        # Placeholder for prediction logic
        pass

    def _split(self, X, y):
        """
        Find the best split for a dataset based on the lowest MSE.
        """
        best_mse = float("inf")
        best_split = None
        n_samples, n_features = X.shape

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])

            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = X[:, feature_index] > threshold

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
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
        """
        Calculate the MSE for a split given left and right target values.
        """
        if len(y_left) == 0 or len(y_right) == 0:
            return float("inf")

        mse_left = np.var(y_left) * len(y_left)
        mse_right = np.var(y_right) * len(y_right)
        total_mse = (mse_left + mse_right) / (len(y_left) + len(y_right))

        return total_mse

    def _build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.
        """
        n_samples, n_features = X.shape
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            leaf_value = self._calculate_leaf_value(y)
            return {"leaf": True, "value": leaf_value}

        split = self._split(X, y)
        if split is None:
            leaf_value = self._calculate_leaf_value(y)
            return {"leaf": True, "value": leaf_value}

        left_tree = self._build_tree(X[split["left_mask"]], y[split["left_mask"]], depth + 1)
        right_tree = self._build_tree(X[split["right_mask"]], y[split["right_mask"]], depth + 1)

        return {
            "leaf": False,
            "feature_index": split["feature_index"],
            "threshold": split["threshold"],
            "left": left_tree,
            "right": right_tree
        }

    def _calculate_leaf_value(self, y):
        """
        Calculate the leaf node prediction value.
        """
        return np.mean(y)

    def _predict_single_input(self, x):
        """
        Traverse the tree to make a prediction for a single input.
        """
        node = self.tree
        while not node["leaf"]:
            if x[node["feature_index"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["value"]


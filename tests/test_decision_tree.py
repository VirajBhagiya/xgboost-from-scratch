from src.decision_tree import DecisionTree
from src.utils import load_data, calculate_rmse

def test_decision_tree_regression():
    X, y = load_data("california_housing")
    tree = DecisionTree(max_depth=3, min_samples_split=10)
    tree.fit(X, y)
    predictions = tree.predict(X)
    rmse = calculate_rmse(y, predictions)
    assert rmse < 10  # Dummy threshold for testing purposes

from src.decision_tree import DecisionTree
from src.utils import load_data, calculate_rmse

def test_decision_tree_regression():
    # Load data
    X, y = load_data("california_housing")
    
    # Test with a shallow tree
    shallow_tree = DecisionTree(max_depth=3, min_samples_split=10)
    shallow_tree.fit(X, y)
    shallow_predictions = shallow_tree.predict(X)
    shallow_rmse = calculate_rmse(y, shallow_predictions)
    assert shallow_rmse < 10, f"RMSE for shallow tree is too high: {shallow_rmse}"

    # Test with a deeper tree
    deep_tree = DecisionTree(max_depth=10, min_samples_split=2)
    deep_tree.fit(X, y)
    deep_predictions = deep_tree.predict(X)
    deep_rmse = calculate_rmse(y, deep_predictions)
    assert deep_rmse < 5, f"RMSE for deep tree is too high: {deep_rmse}"

    # Test with high min_samples_split
    high_min_split_tree = DecisionTree(max_depth=5, min_samples_split=100)
    high_min_split_tree.fit(X, y)
    high_split_predictions = high_min_split_tree.predict(X)
    high_split_rmse = calculate_rmse(y, high_split_predictions)
    assert high_split_rmse < 10, f"RMSE for high min_samples_split tree is too high: {high_split_rmse}"

    print("All DecisionTree regression tests passed.")
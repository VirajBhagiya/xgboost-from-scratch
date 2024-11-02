from src.gradient_boosting import GradientBoosting
from src.utils import load_data, calculate_rmse, calculate_accuracy

def test_gradient_boosting_regression_with_regularization():
    # Regression test with regularization
    X, y = load_data("california_housing")
    model = GradientBoosting(n_estimators=10, learning_rate=0.1, max_depth=3, reg_lambda=1.0, reg_alpha=0.1)
    model.fit(X, y)
    predictions = model.predict(X)
    rmse = calculate_rmse(y, predictions)
    assert rmse < 10, f"RMSE too high with regularization: {rmse}"

def test_gradient_boosting_classification_with_colsample():
    # Classification test with column sampling
    X, y = load_data("iris")
    model = GradientBoosting(n_estimators=10, learning_rate=0.1, max_depth=3, colsample=0.8)
    model.fit(X, y)
    predictions = model.predict(X)
    accuracy = calculate_accuracy(y, predictions)
    assert accuracy > 0.5, f"Accuracy too low with colsample=0.8: {accuracy}"

def test_gradient_boosting_early_stopping():
    # Test early stopping functionality
    X, y = load_data("california_housing")
    model = GradientBoosting(n_estimators=50, learning_rate=0.1, max_depth=3, early_stopping_rounds=5, verbose=True)
    model.fit(X, y)
    predictions = model.predict(X)
    rmse = calculate_rmse(y, predictions)
    assert rmse < 10, f"RMSE too high after early stopping: {rmse}"
    print(f"Best iteration (early stopping): {model.best_iteration}")

def test_gradient_boosting_different_learning_rate():
    # Test model with different learning rates
    X, y = load_data("california_housing")
    model = GradientBoosting(n_estimators=10, learning_rate=0.05, max_depth=3)
    model.fit(X, y)
    predictions = model.predict(X)
    rmse = calculate_rmse(y, predictions)
    assert rmse < 12, f"RMSE too high with learning_rate=0.05: {rmse}"

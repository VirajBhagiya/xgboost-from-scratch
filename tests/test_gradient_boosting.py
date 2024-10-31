from src.gradient_boosting import GradientBoosting
from src.utils import load_data, calculate_rmse, calculate_accuracy

def test_gradient_boosting_with_regularization():
    X, y = load_data("california_housing")
    model = GradientBoosting(n_estimators=10, learning_rate=0.1, max_depth=3, reg_lambda=1.0, reg_alpha=0.1)
    model.fit(X, y)
    predictions = model.predict(X)
    rmse = calculate_rmse(y, predictions)
    assert rmse < 10  # Adjust threshold as needed

def test_gradient_boosting_with_colsample():
    X, y = load_data("iris")
    model = GradientBoosting(n_estimators=10, learning_rate=0.1, max_depth=3, colsample=0.8)
    model.fit(X, y)
    predictions = model.predict(X)
    accuracy = calculate_accuracy(y, predictions)
    assert accuracy > 0.5  # Adjust threshold as needed

import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score

def load_data(dataset_name="california_housing"):
    """
    Load dataset for testing and experimentation.
    """
    from sklearn.datasets import fetch_california_housing, load_iris
    
    if dataset_name == "california_housing":
        data = fetch_california_housing()
    elif dataset_name == "iris":
        data = load_iris()
    else:
        raise ValueError("Unknown dataset.")
    
    X, y = data.data, data.target
    return X, y

def calculate_rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error for regression tasks.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_accuracy(y_true, y_pred):
    """
    Calculate accuracy for classification tasks.
    """
    return accuracy_score(y_true, y_pred)

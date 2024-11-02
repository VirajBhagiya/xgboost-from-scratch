import numpy as np
import time
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.model_selection import KFold

def load_data(dataset_name="california_housing"):
    if dataset_name == "california_housing":
        data = fetch_california_housing()
    elif dataset_name == "iris":
        data = load_iris()
    else:
        raise ValueError("Unknown dataset.")
    
    X, y = data.data, data.target
    return X, y

def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def calculate_precision(y_true, y_pred):
    return precision_score(y_true, y_pred, average='weighted')

def calculate_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average='weighted')

def calculate_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')

def cross_validate(model, X, y, k=5, task='regression', verbose=True, return_all_scores=False):
    kf = KFold(n_splits=k)
    scores = []
    
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        start_time = time.time()
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if verbose:
            print(f"Training fold {fold + 1}/{k}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if task == 'regression':
            score = calculate_rmse(y_test, y_pred)
        else:
            score = calculate_accuracy(y_test, y_pred)
        
        scores.append(score)

        elapsed_time = time.time() - start_time
        if verbose:
            print(f"Fold {fold + 1}, time elapsed: {elapsed_time:.2f} seconds")
            print(f"Score: {score}")

    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    if return_all_scores:
        return mean_score, std_score, scores
    return mean_score, std_score
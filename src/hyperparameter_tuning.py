from itertools import product
from src.gradient_boosting import GradientBoosting
from src.utils import cross_validate

def grid_search(X, y, param_grid, task='regression'):
    best_params = None
    best_score = float('inf') if task == 'regression' else 0
    all_scores = []

    for params in product(*param_grid.values()):
        param_dict = dict(zip(param_grid.keys(), params))
        model = GradientBoosting(**param_dict)
        score_mean, score_std = cross_validate(model, X, y, task=task)

        all_scores.append((param_dict, score_mean, score_std))

        if (task == 'regression' and score_mean < best_score) or (task == 'classification' and score_mean > best_score):
            best_score = score_mean
            best_params = param_dict

    return best_params, best_score, all_scores

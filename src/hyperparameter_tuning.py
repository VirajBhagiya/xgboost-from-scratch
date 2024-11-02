from itertools import product
from src.gradient_boosting import GradientBoosting
from src.utils import cross_validate
from concurrent.futures import ProcessPoolExecutor

def grid_search(X, y, param_grid, task='regression', verbose=False, n_jobs=1):
    best_params = None
    best_score = float('inf') if task == 'regression' else 0
    all_scores = []

    def evaluate(params):
        param_dict = dict(zip(param_grid.keys(), params))
        model = GradientBoosting(**param_dict)
        score_mean, score_std = cross_validate(model, X, y, task=task)
        if verbose:
            print(f"Params: {param_dict}, Score: {score_mean:.4f} ± {score_std:.4f}")
        return param_dict, score_mean, score_std

    if n_jobs == 1:
        results = [evaluate(params) for params in product(*param_grid.values())]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(evaluate, product(*param_grid.values())))

    for param_dict, score_mean, score_std in results:
        all_scores.append((param_dict, score_mean, score_std))
        if (task == 'regression' and score_mean < best_score) or (task == 'classification' and score_mean > best_score):
            best_score = score_mean
            best_params = param_dict

    return best_params, best_score, all_scores
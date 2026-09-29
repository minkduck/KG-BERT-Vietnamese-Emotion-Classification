import numpy as np
from sklearn.metrics import f1_score


def paired_permutation_test(
    y_true, y_pred_a, y_pred_b, metric_func, n_permutations=10000, seed=42
):
    """
    Run non-parametric paired permutation test between two model prediction sets.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred_a (array-like): Predictions from model A.
        y_pred_b (array-like): Predictions from model B.
        metric_func (callable): Performance metric function (e.g., macro_f1).
        n_permutations (int): Number of permutation iterations. Default: 10000.
        seed (int): Random seed for reproducibility. Default: 42.

    Returns:
        tuple: (p_value, observed_difference, score_a, score_b)
    """
    np.random.seed(seed)
    y_true = np.array(y_true)
    y_pred_a = np.array(y_pred_a)
    y_pred_b = np.array(y_pred_b)
    n_samples = len(y_true)

    score_a = metric_func(y_true, y_pred_a)
    score_b = metric_func(y_true, y_pred_b)
    obs_diff = np.abs(score_b - score_a)

    swap_matrix = np.random.rand(n_permutations, n_samples) < 0.5
    count = 0

    for i in range(n_permutations):
        swaps = swap_matrix[i]
        p_a = np.where(swaps, y_pred_b, y_pred_a)
        p_b = np.where(swaps, y_pred_a, y_pred_b)

        s_a = metric_func(y_true, p_a)
        s_b = metric_func(y_true, p_b)
        diff = np.abs(s_b - s_a)

        if diff >= obs_diff:
            count += 1

    p_val = (count + 1) / (n_permutations + 1)
    return p_val, obs_diff, score_a, score_b


def macro_f1(y_true, y_pred):
    """Calculate Macro-averaged F1 Score."""
    return f1_score(y_true, y_pred, average="macro")

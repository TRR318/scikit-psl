"""
This loss is very similar to accuracy, but it is implemented using predict-proba to detect if there are ties among the most probable class.
if so, its assumed that all tied classes will be predicted and a loss will only be incured it the true class is not among those argmax classes
"""
import numpy as np


def ambiguity_aware_accuracy(y_true, y_prob, alpha=0.5):
    """
    Computes a weighted accuracy score inversely proportional to the number 
    of selected classes, modulated by alpha.
    
    Parameters:
    - y_true: np.ndarray of shape (n_samples,), true class indices.
    - y_prob: np.ndarray of shape (n_samples, n_classes), predicted probabilities.
    - alpha: float, weighting factor for ambiguous cases (0 to 1).
    
    Returns:
    - float, the mean of the computed metric.
    """
    # Determine the selected predictions
    if len(y_prob.shape) == 1:
        y_prob = np.vstack((np.ones_like(y_prob) - y_prob, y_prob)).T
    y_pred = y_prob == np.max(y_prob, axis=1, keepdims=True)

    # One-hot encode y_true
    true_one_hot = np.zeros_like(y_prob, dtype=bool)
    true_one_hot[np.arange(len(y_true)), y_true] = True

    # Cases where the true label is selected
    true_selected = y_pred[np.arange(len(y_true)), y_true]

    # Count the number of selected classes for each row
    selected_count = y_pred.sum(axis=1)

    # Calculate the metric
    scores = np.where(
        true_selected,  # If the true label is selected
        alpha * (1 / selected_count) + (1 - alpha) * 1,
        0  # If the true label is not selected
    )

    return scores.mean()

"""Forecast quality metrics.

auc                   ROC-AUC (sklearn.metrics.roc_auc_score)
balanced_accuracy     (sensitivity + specificity) / 2, for predictions already
                      thresholded at 0.5
bce                   binary cross-entropy, with probabilities clipped to
                      (1e-7, 1 - 1e-7)
skill_vs_persistence  AUC(model) - AUC(persistence) on the same probe set

All functions take 1-D numpy arrays. For multi-channel evaluation flatten first:
y_true = targets.ravel(), y_score = probs.ravel().
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Area under the ROC curve.

    Parameters
    ----------
    y_true  : np.ndarray (N,) int or float  binary ground-truth labels {0, 1}
    y_score : np.ndarray (N,) float          predicted probability in [0, 1]

    Returns
    -------
    float  AUC in [0, 1]; 0.5 = chance
    """
    return float(roc_auc_score(y_true, y_score))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Balanced accuracy = (sensitivity + specificity) / 2.

    Equivalent to sklearn balanced_accuracy_score with the default adjusted=False.
    Threshold at 0.5 is assumed to be already applied in y_pred.

    Parameters
    ----------
    y_true : np.ndarray (N,) int or float  binary ground-truth labels {0, 1}
    y_pred : np.ndarray (N,) int or float  binary predictions {0, 1}

    Returns
    -------
    float  balanced accuracy in [0, 1]; 0.5 = chance
    """
    return float(balanced_accuracy_score(y_true.astype(int), y_pred.astype(int)))


def bce(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Binary cross-entropy loss.

    Parameters
    ----------
    y_true : np.ndarray (N,) float  binary ground-truth labels {0.0, 1.0}
    y_prob : np.ndarray (N,) float  predicted probabilities in [0, 1]

    Returns
    -------
    float  mean BCE in [0, inf); 0 = perfect
    """
    y_prob = np.clip(y_prob, 1e-7, 1.0 - 1e-7)
    return float(-np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob)))


def skill_vs_persistence(
    y_true: np.ndarray,
    y_score_model: np.ndarray,
    y_score_persistence: np.ndarray,
) -> float:
    """Skill relative to a reference predictor: AUC(model) - AUC(reference).

    Positive values mean the model beats the reference (persistence, or any other
    baseline passed in its place) on the same probe set.

    Parameters
    ----------
    y_true              : np.ndarray (N,) binary ground-truth labels
    y_score_model       : np.ndarray (N,) model predicted probabilities
    y_score_persistence : np.ndarray (N,) reference predicted probabilities

    Returns
    -------
    float  AUC difference; positive = model beats the reference
    """
    return auc(y_true, y_score_model) - auc(y_true, y_score_persistence)

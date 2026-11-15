import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def binary_metrics(y, probability, threshold=0.5):
    y, p = np.asarray(y), np.asarray(probability)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError('Нужны непустые одномерные y и probability одинаковой длины')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all():
        raise ValueError('Метки должны быть 0/1, вероятности — конечными')
    if ((p < 0) | (p > 1)).any() or not 0 <= threshold <= 1:
        raise ValueError('Вероятности и порог должны быть от 0 до 1')
    predicted = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        'Precision': precision_score(y, predicted, zero_division=0),
        'Recall': recall_score(y, predicted, zero_division=0),
        'F1': f1_score(y, predicted, zero_division=0),
        'AP': average_precision_score(y, p) if y.sum() else np.nan,
        'ROC_AUC': roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
        'Threshold': float(threshold),
    }


def best_threshold(y, probability, grid):

    grid = list(grid)
    if not grid:
        raise ValueError('Сетка порогов пустая')
    binary_metrics(y, probability, grid[0])
    if not np.isfinite(grid).all() or min(grid) < 0 or max(grid) > 1:
        raise ValueError('Пороги должны быть от 0 до 1')
    p = np.asarray(probability)
    return float(max(grid, key=lambda t: f1_score(y, p >= t, zero_division=0)))

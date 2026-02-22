#!/usr/bin/env python3
"""
Module to calculate precision for each class
"""
import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix.
    confusion: (classes, classes) ndarray, columns are predicted labels
    Returns: (classes,) ndarray containing precision of each class
    """
    # Diaqonal elementlər: True Positives (TP)
    tp = np.diag(confusion)

    # Sütun cəmləri: Predicted Positives (TP + FP)
    predicted_positives = np.sum(confusion, axis=0)

    # Precision = TP / (TP + FP)
    return tp / predicted_positives

#!/usr/bin/env python3
"""
Module to calculate sensitivity for each class
"""
import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix.

    confusion: a confusion numpy.ndarray of shape (classes, classes)
               where row indices represent the correct labels and
               column indices represent the predicted labels

    Returns: a numpy.ndarray of shape (classes,) containing the
             sensitivity of each class
    """
    # 1. Diaqonal elementləri (True Positives) götürürük
    tp = np.diag(confusion)

    # 2. Hər bir sətrin cəmini (Həqiqi müsbət elementlərin sayı) tapırıq
    # axis=1 sətir üzrə cəmləmə deməkdir
    actual_positives = np.sum(confusion, axis=1)

    # 3. Sensitivity = TP / Actual Positives
    # Nəticə hər klass üçün hesablanmış bir massiv olacaq
    res = tp / actual_positives

    return res

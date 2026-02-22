#!/usr/bin/env python3
"""Module to calculate specificity"""
import numpy as np


def specificity(confusion):
    """Calculates the specificity for each class in a confusion matrix"""
    # Cəmi elementlərin sayı
    total = np.sum(confusion)
    # Hər sinif üçün TP
    tp = np.diag(confusion)
    # Sətir cəmləri (TP + FN)
    rows = np.sum(confusion, axis=1)
    # Sütun cəmləri (TP + FP)
    cols = np.sum(confusion, axis=0)

    # TN = Ümumi - (Sətir cəmi + Sütun cəmi - TP)
    tn = total - (rows + cols - tp)
    # FP = Sütun cəmi - TP
    fp = cols - tp

    # Specificity = TN / (TN + FP)
    return tn / (tn + fp)

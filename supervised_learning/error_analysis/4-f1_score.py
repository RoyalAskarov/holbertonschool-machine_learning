#!/usr/bin/env python3
"""Module to calculate F1 score for each class"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """Calculates the F1 score for each class in a confusion matrix"""
    # Əvvəlki funksiyalardan istifadə edərək dəyərləri alırıq
    prec = precision(confusion)
    sens = sensitivity(confusion)

    # F1 Score düsturu: 2 * (P * S) / (P + S)
    f1 = 2 * (prec * sens) / (prec + sens)

    return f1

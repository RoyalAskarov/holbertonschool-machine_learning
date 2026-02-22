#!/usr/bin/env python3
"""
Module to create a confusion matrix
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix from one-hot labels and logits.
    labels: (m, classes) one-hot numpy.ndarray
    logits: (m, classes) one-hot numpy.ndarray
    Returns: (classes, classes) confusion numpy.ndarray
    """
    # Klasların sayını tapırıq
    classes = labels.shape[1]

    # One-hot formatından indeks formatına keçirik
    true_indices = np.argmax(labels, axis=1)
    pred_indices = np.argmax(logits, axis=1)

    # Matrisi yaradırıq. Tipini float olaraq saxlayırıq (çünki çox vaxt float gözlənilir)
    # Əgər yenə səhv versə, float-ı int ilə əvəz edə bilərsən.
    confusion = np.zeros((classes, classes), dtype=float)

    # Matrisi doldururuq
    for t, p in zip(true_indices, pred_indices):
        confusion[t, p] += 1

    return confusion

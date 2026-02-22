#!/usr/bin/env python3
import numpy as np
"""
documented module
"""


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix from one-hot labels and logits.
    """
    # 1. Convert one-hot to 1D arrays of class indices
    # (m, classes) -> (m,)
    true_indices = np.argmax(labels, axis=1)
    pred_indices = np.argmax(logits, axis=1)

    # 2. Get the number of classes
    classes = labels.shape[1]

    # 3. Initialize a square matrix of zeros
    confusion = np.zeros((classes, classes))

    # 4. Fill the matrix
    # For each pair of (true, pred), increment the corresponding cell
    for t, p in zip(true_indices, pred_indices):
        confusion[t, p] += 1

    return confusion

#!/usr/bin/env python3
"""
Module to create a confusion matrix
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix from one-hot labels and logits.
    """
    # Convert one-hot to 1D arrays of class indices
    true_indices = np.argmax(labels, axis=1)
    pred_indices = np.argmax(logits, axis=1)

    # Get the number of classes
    classes = labels.shape[1]

    # Initialize a square matrix of zeros
    confusion = np.zeros((classes, classes))

    # Fill the matrix: for each pair, increment the cell
    for t, p in zip(true_indices, pred_indices):
        confusion[t, p] += 1

    return confusion

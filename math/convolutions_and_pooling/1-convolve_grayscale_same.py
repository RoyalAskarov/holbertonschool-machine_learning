#!/usr/bin/env python3
"""
Module to perform a 'same' convolution on grayscale images
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a 'same' convolution on grayscale images

    Args:
        images: numpy.ndarray with shape (m, h, w) containing multiple images
        kernel: numpy.ndarray with shape (kh, kw) containing the kernel

    Returns:
        A numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate padding needed to keep dimensions the same
    # Use // 2 for integer division (works for odd kernel sizes)
    ph = kh // 2
    pw = kw // 2

    # Apply zero padding to the height and width dimensions
    # np.pad(array, ((batch), (height), (width)), mode)
    images_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')

    # Initialize output with original dimensions
    convolved = np.zeros((m, h, w))

    # Perform the convolution (sliding over the padded image)
    for i in range(h):
        for j in range(w):
            # Slice the padded images and perform element-wise multiplication
            image_slice = images_padded[:, i:i + kh, j:j + kw]
            convolved[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2))

    return convolved

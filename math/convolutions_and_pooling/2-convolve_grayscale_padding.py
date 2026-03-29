#!/usr/bin/env python3
"""
Module to perform a convolution on grayscale images with custom padding
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding

    Args:
        images: numpy.ndarray with shape (m, h, w) containing multiple images
        kernel: numpy.ndarray with shape (kh, kw) containing the kernel
        padding: tuple of (ph, pw)
            ph: padding for the height of the image
            pw: padding for the width of the image

    Returns:
        A numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # Apply custom zero padding to height and width
    # Wrap dimensions to keep lines under 79 characters (PEP 8)
    images_padded = np.pad(
        images, ((0, 0), (ph, ph), (pw, pw)), mode='constant'
    )

    # Calculate new output dimensions after padding
    out_h = h + (2 * ph) - kh + 1
    out_w = w + (2 * pw) - kw + 1

    # Initialize the output array
    convolved = np.zeros((m, out_h, out_w))

    # Perform convolution using two loops over output dimensions
    for i in range(out_h):
        for j in range(out_w):
            # Slice the padded images and multiply by the kernel
            image_slice = images_padded[:, i:i + kh, j:j + kw]
            convolved[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2))

    return convolved

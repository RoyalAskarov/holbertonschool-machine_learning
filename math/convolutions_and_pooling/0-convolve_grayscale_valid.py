#!/usr/bin/env python3
"""
Module to perform a valid convolution on grayscale images
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images

    Args:
        images: numpy.ndarray with shape (m, h, w) containing multiple images
            m: number of images
            h: height in pixels of the images
            w: width in pixels of the images
        kernel: numpy.ndarray with shape (kh, kw) containing the kernel
            kh: height of the kernel
            kw: width of the kernel

    Returns:
        A numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate output dimensions
    out_h = h - kh + 1
    out_w = w - kw + 1

    # Initialize the output array with zeros
    convolved = np.zeros((m, out_h, out_w))

    # Iterate over the height and width of the output
    for i in range(out_h):
        for j in range(out_w):
            # Extract the image slice and multiply element-wise with the kernel
            # Then sum over the last two axes (height and width of the slice)
            image_slice = images[:, i:i + kh, j:j + kw]
            convolved[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2))

    return convolved

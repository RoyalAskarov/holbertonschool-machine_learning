#!/usr/bin/env python3
"""
Module to perform pooling on images
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images

    Args:
        images: numpy.ndarray with shape (m, h, w, c)
        kernel_shape: tuple of (kh, kw) containing the kernel shape
        stride: tuple of (sh, sw)
        mode: type of pooling ('max' or 'avg')

    Returns:
        A numpy.ndarray containing the pooled images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Calculate output dimensions
    # Pooling is always "valid" (no padding unless specified)
    out_h = ((h - kh) // sh) + 1
    out_w = ((w - kw) // sw) + 1

    # Initialize output array
    pooled = np.zeros((m, out_h, out_w, c))

    # Iterate over output dimensions
    for i in range(out_h):
        for j in range(out_w):
            # Calculate start positions
            h_start, w_start = i * sh, j * sw
            # Slice current window: (m, kh, kw, c)
            image_slice = images[
                :,
                h_start:h_start + kh,
                w_start:w_start + kw,
                :
            ]

            # Perform operation based on mode
            if mode == 'max':
                # Max over height and width axes (1 and 2)
                pooled[:, i, j, :] = np.max(image_slice, axis=(1, 2))
            elif mode == 'avg':
                # Average over height and width axes (1 and 2)
                pooled[:, i, j, :] = np.mean(image_slice, axis=(1, 2))

    return pooled

#!/usr/bin/env python3
"""
Module to perform a convolution on images with channels
"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with channels

    Args:
        images: numpy.ndarray with shape (m, h, w, c)
        kernel: numpy.ndarray with shape (kh, kw, c)
        padding: 'same', 'valid', or tuple (ph, pw)
        stride: tuple (sh, sw)

    Returns:
        A numpy.ndarray containing the convolved images
    """
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + (1 if kh % 2 == 0 else 0)
        pw = ((w - 1) * sw + kw - w) // 2 + (1 if kw % 2 == 0 else 0)
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # Apply padding only to height and width, not m or c
    images_padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    # Calculate output dimensions
    out_h = ((h + 2 * ph - kh) // sh) + 1
    out_w = ((w + 2 * pw - kw) // sw) + 1

    # Initialize output array (shape: m, out_h, out_w)
    convolved = np.zeros((m, out_h, out_w))

    # Perform convolution
    for i in range(out_h):
        for j in range(out_w):
            h_start, w_start = i * sh, j * sw
            # Extract slice across all channels: (m, kh, kw, c)
            image_slice = images_padded[
                :,
                h_start:h_start + kh,
                w_start:w_start + kw,
                :
            ]
            # Element-wise multiply and sum across axes 1, 2, and 3
            convolved[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2, 3))

    return convolved

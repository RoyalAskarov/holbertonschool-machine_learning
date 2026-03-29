#!/usr/bin/env python3
"""
Module to perform a convolution on images using multiple kernels
"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using multiple kernels
    """
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + (kh % 2 == 0)
        pw = ((w - 1) * sw + kw - w) // 2 + (kw % 2 == 0)
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # Apply padding
    images_padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    # Calculate output dimensions
    out_h = ((h + 2 * ph - kh) // sh) + 1
    out_w = ((w + 2 * pw - kw) // sw) + 1

    # Initialize output array
    convolved = np.zeros((m, out_h, out_w, nc))

    # Perform convolution
    for i in range(out_h):
        for j in range(out_w):
            h_start, w_start = i * sh, j * sw
            # Slice window
            image_slice = images_padded[
                :,
                h_start:h_start + kh,
                w_start:w_start + kw,
                :
            ]

            # Loop through each kernel
            for k in range(nc):
                # Sum across height, width, and channels
                convolved[:, i, j, k] = np.sum(
                    image_slice * kernels[..., k],
                    axis=(1, 2, 3)
                )

    return convolved

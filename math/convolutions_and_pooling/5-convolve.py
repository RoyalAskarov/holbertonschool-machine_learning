#!/usr/bin/env python3
"""
Module to perform a convolution on images using multiple kernels
"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using multiple kernels

    Args:
        images: numpy.ndarray with shape (m, h, w, c)
        kernels: numpy.ndarray with shape (kh, kw, c, nc)
        padding: 'same', 'valid', or tuple (ph, pw)
        stride: tuple (sh, sw)

    Returns:
        A numpy.ndarray containing the convolved images
    """
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + (1 if kh % 2 == 0 else 0)
        pw = ((w - 1) * sw + kw - w) // 2 + (1 if kw % 2 == 0 else 0)
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # Apply padding to height and width
    images_padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    # Calculate output dimensions
    out_h = ((h + 2 * ph - kh) // sh) + 1
    out_w = ((w + 2 * pw - kw) // sw) + 1

    # Initialize output array (m, out_h, out_w, nc)
    convolved = np.zeros((m, out_h, out_w, nc))

    # Perform convolution
    for i in range(out_h):
        for j in range(out_w):
            h_start, w_start = i * sh, j * sw
            # Slice current window across all images and channels
            image_slice = images_padded[
                :,
                h_start:h_start + kh,
                w_start:w_start + kw,
                :
            ]

            # Loop through each kernel (3rd allowed loop)
            for k in range(nc):
                # Multiply slice by k-th kernel and sum across m, h, w, c
                # axis=(1, 2, 3) reduces it to a vector of length m
                convolved[:, i, j, k] = np.sum(
                    image_slice * kernels[..., k],
                    axis=(1, 2, 3)
                )

    return convolved
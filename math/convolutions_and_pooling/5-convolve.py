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

    assert c == kc, "Channels must match"

    # Padding
    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # Pad images
    images_padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    # Output dimensions
    h_out = (h + 2 * ph - kh) // sh + 1
    w_out = (w + 2 * pw - kw) // sw + 1

    output = np.zeros((m, h_out, w_out, nc))

    # Convolution (ONLY 3 loops)
    for i in range(h_out):
        for j in range(w_out):
            for k in range(nc):
                h_start = i * sh
                h_end = h_start + kh
                w_start = j * sw
                w_end = w_start + kw

                slice_img = images_padded[:, h_start:h_end, w_start:w_end, :]
                kernel = kernels[:, :, :, k]

                output[:, i, j, k] = np.sum(slice_img * kernel, axis=(1, 2, 3))

    return output

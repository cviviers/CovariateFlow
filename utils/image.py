import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift


def compute_nps(image, roi_size):
    """
    Compute the Noise-Power Spectrum of an image.

    :param image: 2D numpy array representing the image.
    :param roi_size: Size of the Region of Interest (ROI) as a tuple (height, width).
    :return: 2D NPS array.
    """

    # Extract ROI from the image
    height, width = roi_size
    y, x = image.shape
    startx = x // 2 - (width // 2)
    starty = y // 2 - (height // 2)    
    roi = image[starty:starty+height, startx:startx+width]

    # Apply window function to minimize edge effects
    window = np.hanning(height)[:, None] * np.hanning(width)
    roi_windowed = roi * window

    # Compute the Fourier Transform
    fft_result = fft2(roi_windowed)
    fft_magnitude = np.abs(fftshift(fft_result))

    # Compute the Power Spectrum
    power_spectrum = np.square(fft_magnitude)

    # Normalize the Power Spectrum
    nps = power_spectrum / np.sum(power_spectrum)

    return nps
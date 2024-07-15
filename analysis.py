import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.optimize import linear_sum_assignment


def plot_PSD(signal, fs, n):
    fft_signal = np.fft.fft(signal)
    fft_freqs = np.fft.fftfreq(n, 1/fs)

    # Compute the Power Spectral Density (PSD)
    psd = np.abs(fft_signal) ** 2 / n

    # Only take the positive frequencies
    positive_freqs = fft_freqs[:n // 2]
    positive_psd = psd[:n // 2]

    # Plot the PSD
    plt.figure(figsize=(10, 6))
    plt.plot(positive_freqs, positive_psd)
    plt.title('Power Spectral Density (PSD)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (V^2/Hz)')
    plt.grid()
    plt.show()


# Function to align two arrays and find the optimal position of the smaller array
def align_arrays_with_position(array1, array2):
    """
    Align array2 to array1 using pairwise distances and the Hungarian algorithm,
    and determine the optimal position of array2 in array1.

    Parameters:
    array1 (numpy.ndarray): The larger array of shape (x1, 3).
    array2 (numpy.ndarray): The smaller array of shape (x2, 3).

    Returns:
    tuple: (aligned_array, start_index, end_index)
        - aligned_array (numpy.ndarray): The aligned version of array2 of shape (x1, 3) with NaNs for unmatched elements.
        - start_index (int): The index in array1 where the alignment starts.
        - end_index (int): The index in array1 where the alignment ends.
    """
    x1 = array1.shape[0]
    x2 = array2.shape[0]
    
    # Calculate pairwise Euclidean distances for all possible alignments
    min_cost = np.inf
    best_start = 0

    for start in range(x1 - x2 + 1):
        sub_array1 = array1[start:start + x2]
        distances = np.linalg.norm(sub_array1[:, np.newaxis] - array2, axis=2)
        row_ind, col_ind = linear_sum_assignment(distances)
        cost = distances[row_ind, col_ind].sum()
        
        if cost < min_cost:
            min_cost = cost
            best_start = start
    
    best_end = best_start + x2
    aligned_array = np.full(array1.shape, np.nan)
    aligned_array[best_start:best_end] = array2
    
    return aligned_array, best_start, best_end



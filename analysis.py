import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.optimize import linear_sum_assignment

from osl_dynamics.analysis.spectral import wavelet


def add_noise(signal, SNR=3):

    absolute_signal = np.abs(signal)
    average_amplitude = np.mean(absolute_signal)

    sigma = (average_amplitude / np.sqrt(2)) / SNR
    noise = np.random.normal(0, sigma, signal.shape)
    noisy_signal = signal + noise

    return noisy_signal

def plot_wavelet(data, sampling_frequency, save_name):
    t, f, wt = wavelet(data, sampling_frequency)

    #print(t.shape, f.shape, wt.shape)
    #print(f)
    fig, ax = plt.subplots(figsize=(15, 8))
    cax = ax.imshow(wt[:,0:2000], aspect='auto', cmap='viridis') 
    fig.colorbar(cax, ax = ax, label='Value')  # Show color scale
    ax.set_title('Wavelet')
    ax.set_xlabel('Timepoints')
    ax.set_ylabel('Frequency (Hz)')
       
    data = wt[:,0:2000]
    specific_values = range(100, 2000, 100)
    x_ticks_indices = range(100, 2000, 100)#[np.argmin(np.abs(t - val)) for val in specific_values]
    print(x_ticks_indices)
    
    ax.set_xticks(x_ticks_indices)#, fontsize=8)  

    #ax.set_xticks(np.linspace(0, data.shape[1], 10), np.round(np.linspace(0, data.shape[1], 10), 2))
    
    specific_values = [10, 20, 30, 40]
    y_ticks_indices = [np.argmin(np.abs(f - val)) for val in specific_values]
    ax.set_yticks(y_ticks_indices, np.round(f[y_ticks_indices], 2))#, fontsize=8)  

    ax.grid(False)  # Disable gridlines from seaborn style
    ax.set_facecolor('none') 
    plt.savefig('Wavelet_{}.eps'.format(save_name), format='eps')
    plt.show()    

def plot_PSD1(signal, fs, n):
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

def plot_PSD(x, fs, n, nperseg=4*256, save_name=""):#, noverlap=128):
    
    f, Pxx_den = welch(x, fs, nperseg=nperseg)#, noverlap=noverlap)
    plt.semilogy(f, Pxx_den)
    #plt.ylim([0.5e-3, 1])
    plt.title('Power Spectral Density (PSD)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (V^2/Hz)')
    plt.savefig('PSD_{}.eps'.format(save_name), format='eps')
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



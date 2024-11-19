#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 00:27:27 2023
@author: naveen
"""

import scipy.io as spio
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.ndimage import convolve1d

# 1. Median Filter
def median_filter_1d(data, kernel_size):
    return np.median([np.roll(data, i) for i in range(-kernel_size // 2, kernel_size // 2 + 1)], axis=0)

# 2. Sobel Filter
def sobel_filter_1d(data):
    sobel_kernel = np.array([-1, 0, 1])
    return convolve1d(data, sobel_kernel, mode='nearest')

# 3. Laplacian Filter
def laplacian_filter_1d(data):
    laplacian_kernel = np.array([1, -2, 1])
    return convolve1d(data, laplacian_kernel, mode='nearest')

# 4. High-Pass and Low-Pass Filters
def high_pass_filter_1d(data):
    high_pass_kernel = np.array([-1, -1, 0, 1, 1]) / 2
    return convolve1d(data, high_pass_kernel, mode='nearest')

def low_pass_filter_1d(data):
    low_pass_kernel = np.ones(5) / 5
    return convolve1d(data, low_pass_kernel, mode='nearest')

# 5. Wiener Filter
def wiener_filter_1d(data, noise_var, signal_var):
    return data / (1 + noise_var / signal_var)

# 6. Bilateral Filter
def bilateral_filter_1d(data, sigma_s, sigma_i):
    size = int(6 * sigma_s + 1)
    size += 1 if size % 2 == 0 else 0
    weights = np.exp(-(np.arange(size) - size // 2)**2 / (2 * sigma_s**2))
    weights /= np.sum(weights)
    return np.sum([w * np.roll(data, i) for i, w in enumerate(weights)], axis=0)

# 7. Histogram Equalization
def histogram_equalization_1d(data):
    hist, bins = np.histogram(data, bins=256, range=(0, 1), density=True)
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]
    equalized_data = np.interp(data, bins[:-1], cdf_normalized)
    return equalized_data

# 8. Morphological Filters (Dilation and Erosion)
def dilation_1d(data, kernel_size):
    return np.maximum.reduce([np.roll(data, i) for i in range(-kernel_size // 2, kernel_size // 2 + 1)])

def erosion_1d(data, kernel_size):
    return np.minimum.reduce([np.roll(data, i) for i in range(-kernel_size // 2, kernel_size // 2 + 1)])




def apply_oscillation_correction(data,dt, order=1):
    # Apply Butterworth filter for oscillation correction
    b, a = butter(order, 100*dt, btype='low', analog=False)
    corrected_data = filtfilt(b, a, data)
    return corrected_data

def gaussian_filter_1d(data, sigma):
    """
    Apply a 1D Gaussian filter to the input data.

    Parameters:
    - data: 1D array-like input data
    - sigma: Standard deviation of the Gaussian distribution

    Returns:
    - filtered_data: Result of applying the Gaussian filter
    """
    size = int(6 * sigma + 1)  # Determine the size of the filter kernel
    size += 1 if size % 2 == 0 else 0  # Ensure size is odd

    kernel = np.exp(-(np.arange(size) - size // 2)**2 / (2 * sigma**2))
    kernel /= np.sum(kernel)  # Normalize the kernel to ensure the sum is 1

    filtered_data = np.convolve(data, kernel, mode='same')

    return filtered_data

def plot_data(x, y, exp_x, exp_y,p_history_corrected):
    # Plot pressure over time
    plt.figure(figsize=(8, 6))
    #plt.title(f"Comparison between Numerical and Experimental data at location {probe_location[position]} m")
    plt.xlabel('Time (s)')
    plt.ylabel('Gauge Pressure (Pa)')
    plt.plot(x, y, color="blue", label='Python Simulation')
    plt.plot(exp_x[:], exp_y[:], color="red", linestyle='dashdot', label='Experimental Data')
    #plt.plot(x, p_history_corrected, color="black", label='Python Simulation Filtered')
    # plt.plot(x[times_at_probe_tail], y[times_at_probe_tail], color="green", label='Python Simulation tail')
    plt.legend()
    plt.show()
    plt.savefig('Single_Train_Experiment.png', format='png', dpi=600, bbox_inches='tight')
    

if __name__ == "__main__":
    simulation_data=spio.loadmat('../output/p_history_C.mat')
    p_history = (simulation_data['p_history'])
    u_history=  (simulation_data['u_history'])
    
    position=1
    train_no = 2
    
    if position==0:
        data_loc='../output/experiment/P1.mat'
        mat = spio.loadmat(data_loc)
        exp_x = mat['xdata'][train_no][:]
        exp_y = mat['ydata'][train_no][:]
    elif position==1:
        data_loc='../output/experiment/P3.mat'
        mat = spio.loadmat(data_loc)
        exp_x = mat['xdata'][train_no][:]
        exp_y = mat['ydata'][train_no][:]
    elif position==2:
        data_loc='../output/experiment/two_train.mat'
        mat = spio.loadmat(data_loc)
        exp_x = 3+mat['xdata'][:][0]
        exp_y = mat['ydata'][:][0]
        
    

    
    probe_location=simulation_data['x_probe'][0]
    t=simulation_data['t'][0]


    #p_history_corrected=apply_oscillation_correction(p_history[position],t[1])
    p_history_corrected=gaussian_filter_1d(p_history[position],3)
    



    plot_data(t, p_history[position], exp_x, exp_y,p_history_corrected)
    
    plt.figure(figsize=(8, 6))
    plt.title(f"Air flow data at location {probe_location[position]} m")
    plt.xlabel('Time (s)')
    plt.ylabel('Fluid Velocity (m/s)')
    plt.plot(t, u_history[position], color="blue", label='Python Simulation')
    
    
    """# Example usage:
    # Perform operations
    signal=p_history[position]
    x=t
    median_filtered_signal = median_filter_1d(signal, kernel_size=5)
    sobel_filtered_signal = sobel_filter_1d(signal)
    laplacian_filtered_signal = laplacian_filter_1d(signal)
    high_pass_filtered_signal = high_pass_filter_1d(signal)
    low_pass_filtered_signal = low_pass_filter_1d(signal)
    wiener_filtered_signal = wiener_filter_1d(signal, noise_var=0.04, signal_var=1.0)
    bilateral_filtered_signal = bilateral_filter_1d(signal, sigma_s=1.0, sigma_i=0.1)
    equalized_signal = histogram_equalization_1d(signal)
    dilated_signal = dilation_1d(signal, kernel_size=5)
    eroded_signal = erosion_1d(signal, kernel_size=5)

    # Plot the results
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 4, 1)
    plt.plot(x, signal, label='Original Signal')
    plt.legend()

    plt.subplot(3, 4, 2)
    plt.plot(x, median_filtered_signal, label='Median Filter')
    plt.legend()

    plt.subplot(3, 4, 3)
    plt.plot(x, sobel_filtered_signal, label='Sobel Filter')
    plt.legend()

    plt.subplot(3, 4, 4)
    plt.plot(x, laplacian_filtered_signal, label='Laplacian Filter')
    plt.legend()

    plt.subplot(3, 4, 5)
    #plt.plot(x, high_pass_filtered_signal, label='High-Pass Filter')
    plt.plot(x, low_pass_filtered_signal, label='Low-Pass Filter')
    plt.legend()

    plt.subplot(3, 4, 6)
    plt.plot(x, wiener_filtered_signal, label='Wiener Filter')
    plt.legend()

    plt.subplot(3, 4, 7)
    plt.plot(x, bilateral_filtered_signal, label='Bilateral Filter')
    plt.legend()

    plt.subplot(3, 4, 8)
    plt.plot(x, equalized_signal, label='Histogram Equalization')
    plt.legend()

    plt.subplot(3, 4, 9)
    plt.plot(x, dilated_signal, label='Dilated Signal')
    plt.legend()

    plt.subplot(3, 4, 10)
    plt.plot(x, eroded_signal, label='Eroded Signal')
    plt.legend()

    plt.tight_layout()
    plt.show()"""



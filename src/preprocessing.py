import numpy as np
import scipy.signal as signal
from scipy.stats import zscore

def load_ecg(file_path):
    """
    Placeholder for loading ECG data.
    In practice, use wfdb.rdrecord inside the data loader.
    """
    pass

def resample_ecg(data, original_fs, target_fs):
    """
    Resample ECG signal to target frequency.
    
    Args:
        data (np.array): ECG signal (samples, leads)
        original_fs (int): Original sampling rate
        target_fs (int): Target sampling rate
        
    Returns:
        np.array: Resampled ECG signal
    """
    if original_fs == target_fs:
        return data
    
    number_of_samples = int(data.shape[0] * target_fs / original_fs)
    resampled_data = signal.resample(data, number_of_samples)
    return resampled_data

def bandpass_filter(data, fs, lowcut=0.5, highcut=50.0, order=4):
    """
    Apply bandpass filter to remove noise.
    
    Args:
        data (np.array): ECG signal (samples, leads)
        fs (int): Sampling frequency
        lowcut (float): Low cutoff frequency
        highcut (float): High cutoff frequency
        order (int): Filter order
        
    Returns:
        np.array: Filtered ECG signal
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    
    # Apply filter along time axis (axis 0)
    filtered_data = signal.filtfilt(b, a, data, axis=0)
    return filtered_data

def normalize_ecg(data):
    """
    Apply Z-score normalization per lead.
    
    Args:
        data (np.array): ECG signal (samples, leads)
        
    Returns:
        np.array: Normalized ECG signal
    """
    # zscore computes mean and std along axis 0 by default, which is correct for (samples, leads)
    return zscore(data, axis=0)

def preprocess_signal(data, fs, target_fs=500):
    """
    Full preprocessing pipeline: Resample -> Filter -> Normalize
    """
    # 1. Resample
    if fs != target_fs:
        data = resample_ecg(data, fs, target_fs)
    
    # 2. Filter
    data = bandpass_filter(data, target_fs)
    
    # 3. Normalize
    data = normalize_ecg(data)
    
    return data

import numpy as np
import torch

class Augmentation:
    def __init__(self, sample_rate=500):
        self.sample_rate = sample_rate

    def baseline_wander(self, signal, amplitude=0.5, freq=0.5):
        """Add low frequency baseline wander"""
        t = np.linspace(0, signal.shape[0]/self.sample_rate, signal.shape[0])
        wander = amplitude * np.sin(2 * np.pi * freq * t)
        # Apply to all channels (or randomize)
        # Using broadcasting: signal (Seq, Channels) + wander (Seq, 1) if reshaped
        return signal + wander[:, np.newaxis]

    def gaussian_noise(self, signal, mean=0, std=0.05):
        """Add Gaussian noise"""
        noise = np.random.normal(mean, std, signal.shape)
        return signal + noise

    def channel_scaling(self, signal, min_scale=0.8, max_scale=1.2):
        """Randomly scale channels independently"""
        scales = np.random.uniform(min_scale, max_scale, size=signal.shape[1])
        return signal * scales

    def time_masking(self, signal, max_mask_len=500):
        """Mask a random segment of the signal (dropout in time)"""
        mask_len = np.random.randint(0, max_mask_len)
        start = np.random.randint(0, signal.shape[0] - mask_len)
        signal[start:start+mask_len, :] = 0
        return signal

    def __call__(self, signal):
        """Apply random augmentations"""
        # Signal shape: (Seq_len, Channels) = (5000, 12)
        
        # 1. Baseline Wander (Prob 0.5)
        if np.random.rand() > 0.5:
            amp = np.random.uniform(0.1, 1.0)
            freq = np.random.uniform(0.1, 0.5)
            signal = self.baseline_wander(signal, amp, freq)
            
        # 2. Gaussian Noise (Prob 0.5)
        if np.random.rand() > 0.5:
            std = np.random.uniform(0.01, 0.1)
            signal = self.gaussian_noise(signal, std=std)
            
        # 3. Channel Scaling (Prob 0.5)
        if np.random.rand() > 0.5:
            signal = self.channel_scaling(signal)
            
        # 4. Time Masking (Prob 0.3)
        if np.random.rand() > 0.7:
            signal = self.time_masking(signal)
            
        return signal.astype(np.float32)

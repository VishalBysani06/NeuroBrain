import numpy as np
from scipy import signal
from scipy.stats import kurtosis, skew
import pandas as pd

class EEGFeatureExtractor:
    def __init__(self, fs=250, window_size=2.0, overlap=0.5):
        self.fs = fs
        self.window_size = float(window_size)
        self.overlap = overlap
        self.samples_per_window = int(fs * self.window_size)
        self.step_size = int(self.samples_per_window * (1 - overlap))
    
    def create_windows(self, eeg_data):
        """Create overlapping windows from EEG data"""
        n_samples = eeg_data.shape[0]
        n_channels = eeg_data.shape[1] if eeg_data.ndim > 1 else 1
        
        if eeg_data.ndim == 1:
            eeg_data = eeg_data.reshape(-1, 1)
        
        n_windows = max(1, (n_samples - self.samples_per_window) // self.step_size + 1)
        
        windows = np.zeros((n_windows, self.samples_per_window, n_channels))
        
        for i in range(n_windows):
            start = i * self.step_size
            end = start + self.samples_per_window
            if end <= n_samples:
                windows[i] = eeg_data[start:end]
        
        return windows
    
    def extract_time_domain_features(self, windows):
        """Extract time-domain features"""
        features = []
        feature_names = []
        
        for ch in range(windows.shape[2]):
            ch_data = windows[:, :, ch]
            
            # Basic statistical features
            features.append(np.mean(ch_data, axis=1))  # Mean
            feature_names.append(f'ch{ch}_mean')
            
            features.append(np.std(ch_data, axis=1))   # Standard deviation
            feature_names.append(f'ch{ch}_std')
            
            features.append(np.var(ch_data, axis=1))   # Variance
            feature_names.append(f'ch{ch}_var')
            
            features.append(np.median(ch_data, axis=1)) # Median
            feature_names.append(f'ch{ch}_median')
            
            features.append(np.ptp(ch_data, axis=1))   # Peak-to-peak
            feature_names.append(f'ch{ch}_ptp')
            
            # Higher order statistics
            features.append(skew(ch_data, axis=1))     # Skewness
            feature_names.append(f'ch{ch}_skew')
            
            features.append(kurtosis(ch_data, axis=1)) # Kurtosis
            feature_names.append(f'ch{ch}_kurtosis')
            
            # Signal energy and power
            features.append(np.sum(ch_data**2, axis=1)) # Energy
            feature_names.append(f'ch{ch}_energy')
            
            # Zero crossing rate
            zcr = np.sum(np.diff(np.sign(ch_data), axis=1) != 0, axis=1)
            features.append(zcr)
            feature_names.append(f'ch{ch}_zcr')
            
            # Mobility (first derivative)
            diff1 = np.diff(ch_data, axis=1)
            mobility = np.std(diff1, axis=1) / np.std(ch_data, axis=1)
            features.append(mobility)
            feature_names.append(f'ch{ch}_mobility')
            
            # Complexity (second derivative)
            diff2 = np.diff(diff1, axis=1)
            complexity = np.std(diff2, axis=1) / np.std(diff1, axis=1)
            features.append(complexity)
            feature_names.append(f'ch{ch}_complexity')
        
        return np.column_stack(features), feature_names
    
    def extract_frequency_domain_features(self, windows):
        """Extract frequency-domain features"""
        features = []
        feature_names = []
        
        # Define frequency bands
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        for ch in range(windows.shape[2]):
            ch_data = windows[:, :, ch]
            
            for window_idx in range(ch_data.shape[0]):
                window = ch_data[window_idx]
                
                # Compute power spectral density
                freqs, psd = signal.welch(window, fs=self.fs, nperseg=min(len(window), 256))
                
                # Extract band powers
                for band_name, (low, high) in bands.items():
                    band_mask = (freqs >= low) & (freqs <= high)
                    if np.any(band_mask):
                        band_power = np.trapz(psd[band_mask], freqs[band_mask])
                        
                        if window_idx == 0:  # Initialize feature arrays
                            if f'ch{ch}_{band_name}_power' not in feature_names:
                                features.append([])
                                feature_names.append(f'ch{ch}_{band_name}_power')
                        
                        # Find the correct feature index
                        feature_idx = feature_names.index(f'ch{ch}_{band_name}_power')
                        if len(features[feature_idx]) == window_idx:
                            features[feature_idx].append(band_power)
                
                # Peak frequency
                peak_freq = freqs[np.argmax(psd)]
                if window_idx == 0:
                    if f'ch{ch}_peak_freq' not in feature_names:
                        features.append([])
                        feature_names.append(f'ch{ch}_peak_freq')
                
                feature_idx = feature_names.index(f'ch{ch}_peak_freq')
                if len(features[feature_idx]) == window_idx:
                    features[feature_idx].append(peak_freq)
                
                # Spectral centroid
                spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
                if window_idx == 0:
                    if f'ch{ch}_spectral_centroid' not in feature_names:
                        features.append([])
                        feature_names.append(f'ch{ch}_spectral_centroid')
                
                feature_idx = feature_names.index(f'ch{ch}_spectral_centroid')
                if len(features[feature_idx]) == window_idx:
                    features[feature_idx].append(spectral_centroid)
        
        # Convert to numpy array
        feature_array = np.column_stack([np.array(f) for f in features])
        
        return feature_array, feature_names
    
    def extract_all_features(self, eeg_data):
        """Extract both time and frequency domain features"""
        # Create windows
        windows = self.create_windows(eeg_data)
        
        # Extract features
        time_features, time_names = self.extract_time_domain_features(windows)
        freq_features, freq_names = self.extract_frequency_domain_features(windows)
        
        # Combine features
        all_features = np.column_stack([time_features, freq_features])
        all_names = time_names + freq_names
        
        return all_features, all_names
    
    def create_feature_dataframe(self, features, feature_names, labels=None):
        """Create a pandas DataFrame with features"""
        df = pd.DataFrame(features, columns=feature_names)
        
        if labels is not None:
            df['label'] = labels
        
        return df

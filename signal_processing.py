import numpy as np
import scipy.io
from scipy.signal import butter, lfilter, welch, spectrogram
from scipy import signal
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class EEGSignalProcessor:
    def __init__(self, fs=250):
        self.fs = fs
        self.nyquist = fs / 2
        
    def load_eeg_data(self, file_path, file_type='auto'):
        """Load EEG data from various formats"""
        try:
            if file_type == 'auto':
                file_type = file_path.split('.')[-1].lower()
            
            if file_type == 'mat':
                return self._load_mat_file(file_path)
            elif file_type == 'csv':
                return self._load_csv_file(file_path)
            elif file_type == 'edf':
                return self._load_edf_file(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
        except Exception as e:
            raise Exception(f"Error loading EEG data: {str(e)}")
    
    def _load_mat_file(self, file_path):
        """Load MATLAB .mat files"""
        mat = scipy.io.loadmat(file_path)
        
        # Handle different MATLAB structures
        if 'data' in mat:
            data_struct = mat['data'][0,0]
            eeg_data = data_struct['X'][0]
            
            if isinstance(eeg_data, np.ndarray) and eeg_data.dtype == object:
                eeg_data = np.vstack([ch.reshape(-1, 1) for ch in eeg_data])
            else:
                eeg_data = eeg_data.reshape(-1, 1) if eeg_data.ndim == 1 else eeg_data
        else:
            # Find the largest numerical array
            for key in mat.keys():
                if not key.startswith('__') and isinstance(mat[key], np.ndarray):
                    eeg_data = mat[key]
                    if eeg_data.shape[0] < eeg_data.shape[1]:
                        eeg_data = eeg_data.T
                    break
            else:
                raise ValueError("No EEG data array found")
        
        return eeg_data
    
    def _load_csv_file(self, file_path):
        """Load CSV files"""
        data = pd.read_csv(file_path)
        return data.values
    
    def _load_edf_file(self, file_path):
        """Load EDF files using MNE"""
        try:
            import mne
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            return raw.get_data().T
        except ImportError:
            raise ImportError("MNE library required for EDF files. Install with: pip install mne")
    
    def validate_data(self, eeg_data):
        """Validate EEG data quality"""
        issues = []
        
        # Check for NaN values
        if np.isnan(eeg_data).any():
            issues.append("Data contains NaN values")
        
        # Check for infinite values
        if np.isinf(eeg_data).any():
            issues.append("Data contains infinite values")
        
        # Check data range (typical EEG range is -100 to 100 µV)
        if np.abs(eeg_data).max() > 1000:
            issues.append("Data values seem unusually large (>1000)")
        
        # Check for flat channels
        flat_channels = np.where(np.std(eeg_data, axis=0) < 1e-6)[0]
        if len(flat_channels) > 0:
            issues.append(f"Flat channels detected: {flat_channels}")
        
        return issues
    
    def remove_artifacts(self, eeg_data, method='zscore', threshold=3):
        """Remove artifacts using various methods"""
        cleaned_data = eeg_data.copy()
        
        if method == 'zscore':
            # Z-score based artifact removal
            from scipy.stats import zscore
            z_scores = np.abs(zscore(cleaned_data, axis=0))
            cleaned_data[z_scores > threshold] = np.nan
            
            # Interpolate NaN values
            for ch in range(cleaned_data.shape[1]):
                mask = np.isnan(cleaned_data[:, ch])
                if mask.any():
                    cleaned_data[mask, ch] = np.interp(
                        np.where(mask)[0],
                        np.where(~mask)[0],
                        cleaned_data[~mask, ch]
                    )
        
        return cleaned_data
    
    def apply_bandpass_filter(self, eeg_data, low_freq=8, high_freq=30, order=5):
        """Apply bandpass filter"""
        try:
            low = low_freq / self.nyquist
            high = high_freq / self.nyquist
            
            if high >= 1.0:
                high = 0.99
            
            b, a = butter(order, [low, high], btype='band')
            filtered_data = lfilter(b, a, eeg_data, axis=0)
            return filtered_data
        except Exception as e:
            raise Exception(f"Filtering failed: {str(e)}")
    
    def extract_frequency_bands(self, eeg_data):
        """Extract different frequency bands"""
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        band_power = {}
        
        for band_name, (low, high) in bands.items():
            if high < self.nyquist:
                filtered = self.apply_bandpass_filter(eeg_data, low, high)
                band_power[band_name] = np.mean(filtered**2, axis=0)
        
        return band_power
    
    def compute_psd(self, eeg_data, nperseg=None):
        """Compute Power Spectral Density"""
        if nperseg is None:
            nperseg = min(self.fs * 2, eeg_data.shape[0] // 4)
        
        freqs, psd = welch(eeg_data, fs=self.fs, nperseg=nperseg, axis=0)
        return freqs, psd
    
    def compute_spectrogram(self, eeg_data, nperseg=None):
        """Compute spectrogram"""
        if nperseg is None:
            nperseg = min(self.fs, eeg_data.shape[0] // 10)
        
        freqs, times, Sxx = spectrogram(
            eeg_data[:, 0] if eeg_data.ndim > 1 else eeg_data,
            fs=self.fs,
            nperseg=nperseg
        )
        return freqs, times, Sxx
    
    def normalize_data(self, eeg_data, method='zscore'):
        """Normalize EEG data"""
        if method == 'zscore':
            return (eeg_data - np.mean(eeg_data, axis=0)) / np.std(eeg_data, axis=0)
        elif method == 'minmax':
            min_val = np.min(eeg_data, axis=0)
            max_val = np.max(eeg_data, axis=0)
            return (eeg_data - min_val) / (max_val - min_val)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

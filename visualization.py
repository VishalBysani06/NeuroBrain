import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import pandas as pd

class EEGVisualizer:
    def __init__(self, fs=250):
        self.fs = fs
        
    def plot_raw_signal(self, eeg_data, channels=None, duration=10):
        """Plot raw EEG signals"""
        if channels is None:
            channels = min(4, eeg_data.shape[1] if eeg_data.ndim > 1 else 1)
        
        if eeg_data.ndim == 1:
            eeg_data = eeg_data.reshape(-1, 1)
        
        # Limit duration
        max_samples = int(duration * self.fs)
        data_to_plot = eeg_data[:max_samples, :channels]
        
        # Create time axis
        time = np.arange(data_to_plot.shape[0]) / self.fs
        
        # Create subplots
        fig = make_subplots(
            rows=channels, cols=1,
            shared_xaxes=True,
            subplot_titles=[f'Channel {i+1}' for i in range(channels)],
            vertical_spacing=0.02
        )
        
        for ch in range(channels):
            fig.add_trace(
                go.Scatter(
                    x=time, 
                    y=data_to_plot[:, ch],
                    mode='lines',
                    name=f'Ch {ch+1}',
                    line=dict(width=1)
                ),
                row=ch+1, col=1
            )
        
        fig.update_layout(
            title="Raw EEG Signals",
            height=200 * channels,
            showlegend=False,
            xaxis_title="Time (s)",
            template="plotly_white"
        )
        
        # Update y-axis labels
        for i in range(channels):
            fig.update_yaxes(title_text="Amplitude (µV)", row=i+1, col=1)
        
        return fig
    
    def plot_power_spectrum(self, eeg_data, freqs, psd, max_freq=50):
        """Plot power spectral density"""
        # Limit frequency range
        freq_mask = freqs <= max_freq
        freqs_plot = freqs[freq_mask]
        
        if eeg_data.ndim == 1 or psd.ndim == 1:
            psd_plot = psd[freq_mask]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=freqs_plot,
                y=10 * np.log10(psd_plot),
                mode='lines',
                name='PSD',
                line=dict(width=2)
            ))
            
        else:
            # Multiple channels
            psd_plot = psd[freq_mask, :]
            n_channels = min(4, psd_plot.shape[1])
            
            fig = go.Figure()
            for ch in range(n_channels):
                fig.add_trace(go.Scatter(
                    x=freqs_plot,
                    y=10 * np.log10(psd_plot[:, ch]),
                    mode='lines',
                    name=f'Channel {ch+1}',
                    line=dict(width=2)
                ))
        
        # Add frequency band markers
        bands = {
            'Delta': (0.5, 4, 'rgba(255, 0, 0, 0.2)'),
            'Theta': (4, 8, 'rgba(255, 165, 0, 0.2)'),
            'Alpha': (8, 12, 'rgba(255, 255, 0, 0.2)'),
            'Beta': (13, 30, 'rgba(0, 255, 0, 0.2)'),
            'Gamma': (30, 50, 'rgba(0, 0, 255, 0.2)')
        }
        
        for band_name, (low, high, color) in bands.items():
            if high <= max_freq:
                fig.add_vrect(
                    x0=low, x1=high,
                    fillcolor=color,
                    opacity=0.3,
                    layer="below",
                    line_width=0,
                    annotation_text=band_name,
                    annotation_position="top left"
                )
        
        fig.update_layout(
            title="Power Spectral Density",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Power (dB)",
            template="plotly_white",
            height=500
        )
        
        return fig
    
    def plot_spectrogram(self, freqs, times, Sxx, max_freq=50):
        """Plot spectrogram"""
        # Limit frequency range
        freq_mask = freqs <= max_freq
        freqs_plot = freqs[freq_mask]
        Sxx_plot = Sxx[freq_mask, :]
        
        fig = go.Figure(data=go.Heatmap(
            z=10 * np.log10(Sxx_plot),
            x=times,
            y=freqs_plot,
            colorscale='Viridis',
            colorbar=dict(title="Power (dB)")
        ))
        
        fig.update_layout(
            title="EEG Spectrogram",
            xaxis_title="Time (s)",
            yaxis_title="Frequency (Hz)",
            template="plotly_white",
            height=500
        )
        
        return fig
    
    def plot_frequency_bands(self, band_powers):
        """Plot frequency band powers"""
        bands = list(band_powers.keys())
        powers = [np.mean(band_powers[band]) for band in bands]
        
        fig = go.Figure(data=[
            go.Bar(x=bands, y=powers, marker_color='steelblue')
        ])
        
        fig.update_layout(
            title="Average Power by Frequency Band",
            xaxis_title="Frequency Bands",
            yaxis_title="Average Power",
            template="plotly_white"
        )
        
        return fig
    
    def plot_feature_importance(self, importance_dict, top_n=20):
        """Plot feature importance"""
        if importance_dict is None:
            return None
        
        # Handle both dict and numpy array inputs
        if isinstance(importance_dict, dict):
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        elif isinstance(importance_dict, np.ndarray):
            # Create feature names if not provided
            feature_names = [f'Feature_{i}' for i in range(len(importance_dict))]
            sorted_features = sorted(zip(feature_names, importance_dict), key=lambda x: x[1], reverse=True)
        else:
            return None
        top_features = sorted_features[:top_n]
        
        features, importances = zip(*top_features)
        
        fig = go.Figure(data=[
            go.Bar(x=list(importances), y=list(features), orientation='h',
                   marker_color='lightblue')
        ])
        
        fig.update_layout(
            title=f"Top {top_n} Feature Importances",
            xaxis_title="Importance",
            yaxis_title="Features",
            template="plotly_white",
            height=600
        )
        
        return fig
    
    def plot_confusion_matrix(self, cm, labels):
        """Plot confusion matrix"""
        fig = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='Blues',
            labels=dict(x="Predicted", y="Actual", color="Count")
        )
        
        fig.update_layout(
            title="Confusion Matrix",
            xaxis=dict(tickmode='array', tickvals=list(range(len(labels))), ticktext=labels),
            yaxis=dict(tickmode='array', tickvals=list(range(len(labels))), ticktext=labels),
            template="plotly_white"
        )
        
        return fig
    
    def plot_prediction_confidence(self, predictions_data):
        """Plot prediction confidence over time"""
        confidences = [p['confidence'] for p in predictions_data]
        predictions = [p['prediction'] for p in predictions_data]
        reliable = [p['reliable'] for p in predictions_data]
        
        fig = go.Figure()
        
        # Plot confidence over time
        fig.add_trace(go.Scatter(
            y=confidences,
            mode='lines+markers',
            name='Confidence',
            line=dict(color='blue', width=2),
            marker=dict(
                color=['green' if r else 'red' for r in reliable],
                size=8
            )
        ))
        
        # Add threshold line
        fig.add_hline(y=0.7, line_dash="dash", line_color="red", 
                      annotation_text="Confidence Threshold")
        
        fig.update_layout(
            title="Prediction Confidence Over Time",
            xaxis_title="Prediction #",
            yaxis_title="Confidence",
            template="plotly_white",
            height=400
        )
        
        return fig
    
    def plot_eeg_topography(self, values, positions=None):
        """Plot EEG topography (simplified version)"""
        if positions is None:
            # Default electrode positions for common montages
            n_channels = len(values)
            if n_channels <= 4:
                positions = [(0, 1), (1, 0), (0, -1), (-1, 0)][:n_channels]
            else:
                # Circular arrangement
                angles = np.linspace(0, 2*np.pi, n_channels, endpoint=False)
                positions = [(np.cos(a), np.sin(a)) for a in angles]
        
        x, y = zip(*positions)
        
        fig = go.Figure(data=go.Scatter(
            x=x, y=y,
            mode='markers+text',
            marker=dict(
                size=30,
                color=values,
                colorscale='RdBu',
                showscale=True,
                colorbar=dict(title="Amplitude")
            ),
            text=[f'Ch{i+1}' for i in range(len(values))],
            textposition="middle center"
        ))
        
        fig.update_layout(
            title="EEG Topography",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template="plotly_white",
            height=500,
            width=500
        )
        
        return fig

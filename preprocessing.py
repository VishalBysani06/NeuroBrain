import streamlit as st
import numpy as np
from utils.signal_processing import EEGSignalProcessor
from utils.visualization import EEGVisualizer
import plotly.graph_objects as go

def render_preprocessing():
    st.header("⚙️ Signal Preprocessing & Enhancement")
    
    if st.session_state.eeg_data is None:
        st.warning("⚠️ Please upload EEG data first in the Data Upload tab.")
        return
    
    processor = EEGSignalProcessor()
    visualizer = EEGVisualizer()
    
    eeg_data = st.session_state.eeg_data.copy()
    
    # Preprocessing options
    st.subheader("Preprocessing Pipeline")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Processing Steps")
        
        # Artifact removal
        remove_artifacts = st.checkbox("Remove Artifacts", value=True)
        if remove_artifacts:
            artifact_method = st.selectbox(
                "Artifact Removal Method",
                options=['zscore', 'iqr'],
                help="Method for detecting and removing artifacts"
            )
            artifact_threshold = st.slider(
                "Artifact Threshold",
                min_value=1.0,
                max_value=5.0,
                value=3.0,
                step=0.5,
                key="preprocessing_artifact_threshold"
            )
        
        # Filtering
        apply_filter = st.checkbox("Apply Bandpass Filter", value=True)
        if apply_filter:
            col_freq1, col_freq2 = st.columns(2)
            with col_freq1:
                low_freq = st.number_input(
                    "Low Freq (Hz)",
                    min_value=0.1,
                    max_value=50.0,
                    value=8.0,
                    step=0.5
                )
            with col_freq2:
                high_freq = st.number_input(
                    "High Freq (Hz)",
                    min_value=1.0,
                    max_value=100.0,
                    value=30.0,
                    step=0.5
                )
            
            filter_order = st.slider(
                "Filter Order",
                min_value=2,
                max_value=10,
                value=5,
                key="preprocessing_filter_order"
            )
        
        # Normalization
        apply_normalization = st.checkbox("Normalize Data", value=True)
        if apply_normalization:
            norm_method = st.selectbox(
                "Normalization Method",
                options=['zscore', 'minmax'],
                help="Method for normalizing the data"
            )
        
        # Process button
        if st.button("Apply Preprocessing", type="primary"):
            with st.spinner("Processing EEG data..."):
                processed_data = eeg_data.copy()
                processing_steps = []
                
                # Remove DC offset (always applied)
                processed_data = processed_data - np.mean(processed_data, axis=0)
                processing_steps.append("✅ DC offset removed")
                
                # Artifact removal
                if remove_artifacts:
                    try:
                        processed_data = processor.remove_artifacts(
                            processed_data, 
                            method=artifact_method,
                            threshold=artifact_threshold
                        )
                        processing_steps.append(f"✅ Artifacts removed ({artifact_method})")
                    except Exception as e:
                        st.error(f"Artifact removal failed: {e}")
                
                # Filtering
                if apply_filter:
                    try:
                        processed_data = processor.apply_bandpass_filter(
                            processed_data,
                            low_freq=low_freq,
                            high_freq=high_freq,
                            order=filter_order
                        )
                        processing_steps.append(f"✅ Bandpass filter applied ({low_freq}-{high_freq} Hz)")
                    except Exception as e:
                        st.error(f"Filtering failed: {e}")
                
                # Normalization
                if apply_normalization:
                    try:
                        processed_data = processor.normalize_data(
                            processed_data,
                            method=norm_method
                        )
                        processing_steps.append(f"✅ Data normalized ({norm_method})")
                    except Exception as e:
                        st.error(f"Normalization failed: {e}")
                
                # Store processed data
                st.session_state.processed_data = processed_data
                st.session_state.processing_steps = processing_steps
                
                st.success("✅ Preprocessing completed successfully!")
    
    with col2:
        st.markdown("### Signal Comparison")
        
        if st.session_state.processed_data is not None:
            # Show processing steps
            st.markdown("**Applied Processing Steps:**")
            for step in st.session_state.processing_steps:
                st.markdown(f"- {step}")
            
            # Signal comparison plot
            processed_data = st.session_state.processed_data
            
            # Select channel to display
            n_channels = eeg_data.shape[1] if eeg_data.ndim > 1 else 1
            if n_channels > 1:
                selected_channel = st.selectbox(
                    "Select Channel for Comparison",
                    options=[f"Channel {i+1}" for i in range(n_channels)],
                    index=0
                )
                ch_idx = int(selected_channel.split()[-1]) - 1
            else:
                ch_idx = 0
            
            # Time axis
            duration = min(10, eeg_data.shape[0] / 250)  # Max 10 seconds
            samples_to_plot = int(duration * 250)
            time_axis = np.arange(samples_to_plot) / 250
            
            # Create comparison plot
            fig = go.Figure()
            
            if eeg_data.ndim > 1:
                original_signal = eeg_data[:samples_to_plot, ch_idx]
                processed_signal = processed_data[:samples_to_plot, ch_idx]
            else:
                original_signal = eeg_data[:samples_to_plot]
                processed_signal = processed_data[:samples_to_plot]
            
            fig.add_trace(go.Scatter(
                x=time_axis,
                y=original_signal,
                mode='lines',
                name='Original',
                line=dict(color='red', width=1.5, dash='dot'),
                opacity=0.7
            ))
            
            fig.add_trace(go.Scatter(
                x=time_axis,
                y=processed_signal,
                mode='lines',
                name='Processed',
                line=dict(color='blue', width=2)
            ))
            
            fig.update_layout(
                title=f"Signal Comparison - {selected_channel if n_channels > 1 else 'EEG Signal'}",
                xaxis_title="Time (s)",
                yaxis_title="Amplitude",
                template="plotly_white",
                height=400,
                legend=dict(x=0.02, y=0.98)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Apply preprocessing to see signal comparison")
    
    # Frequency analysis section
    if st.session_state.processed_data is not None:
        st.subheader("Frequency Domain Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Power spectral density
            st.markdown("### Power Spectral Density")
            
            processed_data = st.session_state.processed_data
            freqs, psd = processor.compute_psd(processed_data)
            
            psd_fig = visualizer.plot_power_spectrum(processed_data, freqs, psd)
            st.plotly_chart(psd_fig, use_container_width=True)
        
        with col2:
            # Frequency bands
            st.markdown("### Frequency Band Analysis")
            
            band_powers = processor.extract_frequency_bands(processed_data)
            
            if band_powers:
                bands_fig = visualizer.plot_frequency_bands(band_powers)
                st.plotly_chart(bands_fig, use_container_width=True)
                
                # Show band power values
                st.markdown("**Band Power Summary:**")
                for band, power in band_powers.items():
                    avg_power = np.mean(power)
                    st.metric(
                        label=f"{band.capitalize()} Band",
                        value=f"{avg_power:.2e}",
                        help=f"Average power in {band} frequency band"
                    )
        
        # Spectrogram
        st.markdown("### Time-Frequency Analysis")
        
        # Compute spectrogram for first channel
        if processed_data.ndim > 1:
            signal_for_spec = processed_data[:, 0]
        else:
            signal_for_spec = processed_data
        
        freqs_spec, times_spec, Sxx = processor.compute_spectrogram(signal_for_spec)
        spec_fig = visualizer.plot_spectrogram(freqs_spec, times_spec, Sxx)
        st.plotly_chart(spec_fig, use_container_width=True)
        
        # Quality metrics
        st.subheader("Signal Quality Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            snr = np.mean(processed_data**2) / np.var(processed_data)
            st.metric("Signal-to-Noise Ratio", f"{snr:.2f} dB")
        
        with col2:
            signal_power = np.mean(processed_data**2)
            st.metric("Signal Power", f"{signal_power:.2e}")
        
        with col3:
            peak_freq = freqs[np.argmax(np.mean(psd, axis=1) if psd.ndim > 1 else psd)]
            st.metric("Peak Frequency", f"{peak_freq:.1f} Hz")
        
        with col4:
            alpha_power = band_powers.get('alpha', [0])
            alpha_dominance = np.mean(alpha_power) / np.sum([np.mean(p) for p in band_powers.values()])
            st.metric("Alpha Dominance", f"{alpha_dominance*100:.1f}%")
        
        # Proceed to next step
        if st.button("Proceed to Feature Extraction", type="primary"):
            st.success("✅ Preprocessing completed! Switch to the Model Training tab to continue.")
    
    else:
        # Instructions when no preprocessing applied
        st.info("""
        ### Preprocessing Pipeline
        
        Signal preprocessing is crucial for accurate EEG analysis. The pipeline includes:
        
        1. **DC Offset Removal**: Removes baseline drift
        2. **Artifact Removal**: Detects and removes noisy segments
        3. **Bandpass Filtering**: Focuses on relevant frequency ranges
        4. **Normalization**: Standardizes signal amplitude
        
        ### Recommended Settings
        - **Motor Imagery**: 8-30 Hz bandpass filter
        - **Imagined Speech**: 8-30 Hz for general analysis
        - **Artifact Threshold**: 3.0 (moderate sensitivity)
        - **Normalization**: Z-score for consistency
        
        Configure your preprocessing parameters and click "Apply Preprocessing" to enhance your EEG signals.
        """)

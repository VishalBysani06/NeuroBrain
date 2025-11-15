import streamlit as st
import numpy as np
import pandas as pd
from utils.signal_processing import EEGSignalProcessor
import plotly.graph_objects as go

def render_data_upload():
    st.header("📁 EEG Data Upload & Validation")
    
    processor = EEGSignalProcessor()
    
    # File upload section
    st.subheader("Upload EEG Data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an EEG file",
            type=['mat', 'csv', 'edf'],
            help="Supported formats: MATLAB (.mat), CSV (.csv), EDF (.edf)"
        )
    
    with col2:
        st.info("""
        **File Requirements:**
        - MATLAB .mat files with EEG data
        - CSV files with time series data
        - EDF files (requires MNE library)
        - Sampling rate: 250 Hz (recommended)
        """)
    
    if uploaded_file is not None:
        try:
            # Save uploaded file temporarily
            with st.spinner("Loading EEG data..."):
                file_path = f"temp_{uploaded_file.name}"
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load and process data
                eeg_data = processor.load_eeg_data(file_path)
                st.session_state.eeg_data = eeg_data
            
            st.success(f"✅ Successfully loaded EEG data!")
            
            # Display data information
            st.subheader("Data Information")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Samples", f"{eeg_data.shape[0]:,}")
            
            with col2:
                st.metric("Channels", eeg_data.shape[1] if eeg_data.ndim > 1 else 1)
            
            with col3:
                duration = eeg_data.shape[0] / 250  # Assuming 250 Hz
                st.metric("Duration", f"{duration:.1f}s")
            
            with col4:
                file_size = uploaded_file.size / (1024 * 1024)
                st.metric("File Size", f"{file_size:.1f} MB")
            
            # Data validation
            st.subheader("Data Quality Assessment")
            
            issues = processor.validate_data(eeg_data)
            
            if not issues:
                st.success("✅ No data quality issues detected!")
            else:
                st.warning("⚠️ Data quality issues detected:")
                for issue in issues:
                    st.write(f"- {issue}")
            
            # Quick visualization
            st.subheader("Quick Data Preview")
            
            # Show basic statistics
            if eeg_data.ndim > 1:
                stats_df = pd.DataFrame({
                    'Channel': [f'Ch {i+1}' for i in range(eeg_data.shape[1])],
                    'Mean': np.mean(eeg_data, axis=0),
                    'Std': np.std(eeg_data, axis=0),
                    'Min': np.min(eeg_data, axis=0),
                    'Max': np.max(eeg_data, axis=0)
                })
            else:
                stats_df = pd.DataFrame({
                    'Metric': ['Mean', 'Std', 'Min', 'Max'],
                    'Value': [np.mean(eeg_data), np.std(eeg_data), 
                             np.min(eeg_data), np.max(eeg_data)]
                })
            
            st.dataframe(stats_df, use_container_width=True)
            
            # Plot preview
            st.subheader("Signal Preview")
            
            # Limit preview to first 10 seconds
            preview_samples = min(2500, eeg_data.shape[0])  # 10 seconds at 250 Hz
            time_axis = np.arange(preview_samples) / 250
            
            fig = go.Figure()
            
            if eeg_data.ndim > 1:
                # Plot first few channels
                n_channels_to_plot = min(4, eeg_data.shape[1])
                for ch in range(n_channels_to_plot):
                    fig.add_trace(go.Scatter(
                        x=time_axis,
                        y=eeg_data[:preview_samples, ch],
                        mode='lines',
                        name=f'Channel {ch+1}',
                        line=dict(width=1)
                    ))
            else:
                fig.add_trace(go.Scatter(
                    x=time_axis,
                    y=eeg_data[:preview_samples],
                    mode='lines',
                    name='EEG Signal',
                    line=dict(width=1, color='blue')
                ))
            
            fig.update_layout(
                title="EEG Signal Preview (First 10 seconds)",
                xaxis_title="Time (s)",
                yaxis_title="Amplitude (µV)",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Configuration options
            st.subheader("Processing Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                sampling_rate = st.number_input(
                    "Sampling Rate (Hz)",
                    min_value=1,
                    max_value=5000,
                    value=250,
                    help="EEG data sampling rate"
                )
                
                window_size = st.slider(
                    "Analysis Window Size (seconds)",
                    min_value=1.0,
                    max_value=5.0,
                    value=2.0,
                    step=0.5,
                    help="Time window for feature extraction",
                    key="upload_window_size"
                )
            
            with col2:
                overlap = st.slider(
                    "Window Overlap (%)",
                    min_value=0,
                    max_value=90,
                    value=50,
                    step=10,
                    help="Overlap between consecutive windows",
                    key="upload_overlap"
                )
                
                intent_type = st.selectbox(
                    "Intent Classification Type",
                    options=['Motor Imagery', 'Imagined Speech'],
                    help="Type of brain activity to classify"
                )
            
            # Store configuration in session state
            st.session_state.config = {
                'sampling_rate': sampling_rate,
                'window_size': window_size,
                'overlap': overlap / 100,
                'intent_type': intent_type.lower().replace(' ', '_')
            }
            
            if st.button("Proceed to Preprocessing", type="primary"):
                st.success("✅ Data loaded successfully! Switch to the Preprocessing tab to continue.")
                
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            st.info("""
            **Troubleshooting Tips:**
            1. Ensure your file contains valid EEG data
            2. Check that the file format is supported
            3. Verify the data structure (samples × channels)
            4. Make sure there are no corrupted values
            """)
    
    else:
        # Instructions when no file is uploaded
        st.info("""
        ### Getting Started
        
        1. **Upload your EEG data file** using the file uploader above
        2. **Supported formats**: .mat (MATLAB), .csv (Comma-separated values), .edf (European Data Format)
        3. **Data structure**: Your file should contain EEG time series data with samples as rows and channels as columns
        4. **Sampling rate**: 250 Hz is recommended for optimal performance
        
        ### Example Data Structure
        ```
        Time (samples) × Channels matrix
        Sample 1: [ch1_val, ch2_val, ch3_val, ...]
        Sample 2: [ch1_val, ch2_val, ch3_val, ...]
        ...
        ```
        
        Start by uploading your EEG data file to begin the analysis pipeline.
        """)
        
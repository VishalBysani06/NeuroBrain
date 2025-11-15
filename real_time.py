import streamlit as st
import numpy as np
import pandas as pd
import time
from utils.ml_models import EEGClassifier
from utils.feature_extraction import EEGFeatureExtractor
from utils.visualization import EEGVisualizer
import plotly.graph_objects as go
from datetime import datetime

def render_real_time():
    st.header("⚡ Real-time EEG Prediction")
    
    if st.session_state.trained_model is None:
        st.warning("⚠️ Please train a model first in the Model Training tab.")
        return
    
    # Initialize session state for real-time data
    if 'real_time_predictions' not in st.session_state:
        st.session_state.real_time_predictions = []
    if 'is_predicting' not in st.session_state:
        st.session_state.is_predicting = False
    
    classifier = st.session_state.trained_model
    
    # Real-time prediction interface
    st.subheader("Real-time Classification Interface")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Prediction Settings")
        
        # Prediction parameters
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.5,
            max_value=0.95,
            value=0.7,
            step=0.05,
            help="Minimum confidence for reliable predictions"
        )
        
        prediction_interval = st.slider(
            "Prediction Interval (seconds)",
            min_value=0.5,
            max_value=5.0,
            value=1.0,
            step=0.5,
            help="Time between consecutive predictions"
        )
        
        # Intent type selection
        intent_type = st.selectbox(
            "Intent Type",
            options=['Motor Imagery', 'Imagined Speech'],
            help="Type of intents to predict"
        )
        
        # Control buttons
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("Start Prediction", type="primary", disabled=st.session_state.is_predicting):
                st.session_state.is_predicting = True
                st.rerun()
        
        with col_b:
            if st.button("Stop Prediction", disabled=not st.session_state.is_predicting):
                st.session_state.is_predicting = False
                st.rerun()
        
        if st.button("Clear History"):
            st.session_state.real_time_predictions = []
            st.rerun()
        
        # Model information
        st.markdown("### Model Information")
        st.info(f"""
        **Model Type:** {classifier.model_type.title()}
        **Training Status:** ✅ Trained
        **Intent Mapping:** {intent_type}
        """)
        
        # Current intent mapping
        if intent_type == 'Motor Imagery':
            intents = classifier.motor_imagery_intents
        else:
            intents = classifier.speech_intents
        
        st.markdown("**Intent Classes:**")
        for key, value in intents.items():
            st.write(f"- {value}")
    
    with col2:
        st.markdown("### Prediction Results")
        
        # Real-time prediction simulation
        if st.session_state.is_predicting:
            # Simulate real-time EEG data processing
            simulate_real_time_prediction(classifier, confidence_threshold, intent_type)
        
        # Display recent predictions
        if st.session_state.real_time_predictions:
            display_prediction_history()
        else:
            st.info("No predictions yet. Start real-time prediction to see results.")
    
    # Visualization section
    if st.session_state.real_time_predictions:
        st.subheader("Real-time Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Confidence over time
            plot_confidence_timeline()
        
        with col2:
            # Intent distribution
            plot_intent_distribution()
        
        # Detailed prediction table
        st.subheader("Prediction History")
        display_prediction_table()

def simulate_real_time_prediction(classifier, confidence_threshold, intent_type):
    """Simulate real-time EEG prediction"""
    
    # Use existing processed data to simulate real-time windows
    if st.session_state.processed_data is not None:
        processed_data = st.session_state.processed_data
        config = getattr(st.session_state, 'config', {})
        
        # Create feature extractor
        extractor = EEGFeatureExtractor(
            fs=config.get('sampling_rate', 250),
            window_size=2.0,
            overlap=0.5
        )
        
        # Create a placeholder for real-time updates
        prediction_placeholder = st.empty()
        
        with prediction_placeholder.container():
            st.markdown("### 🔴 Live Prediction")
            
            # Simulate getting a random window from the data
            n_samples = processed_data.shape[0]
            window_size_samples = int(2.0 * config.get('sampling_rate', 250))
            
            if n_samples > window_size_samples:
                # Random window selection
                start_idx = np.random.randint(0, n_samples - window_size_samples)
                window_data = processed_data[start_idx:start_idx + window_size_samples]
                
                # Extract features
                windows = extractor.create_windows(window_data)
                if windows.shape[0] > 0:
                    if hasattr(extractor, 'extract_all_features'):
                        features, _ = extractor.extract_all_features(window_data)
                    else:
                        time_features, _ = extractor.extract_time_domain_features(windows)
                        freq_features, _ = extractor.extract_frequency_domain_features(windows)
                        features = np.column_stack([time_features, freq_features])
                    
                    # Make prediction
                    if features.shape[0] > 0:
                        predictions = classifier.predict_with_confidence(
                            features[0:1], threshold=confidence_threshold
                        )
                        
                        prediction = predictions[0]
                        
                        # Store prediction
                        st.session_state.real_time_predictions.append({
                            'timestamp': datetime.now(),
                            'prediction': prediction['prediction'],
                            'confidence': prediction['confidence'],
                            'reliable': prediction['reliable']
                        })
                        
                        # Display current prediction
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            status_color = "🟢" if prediction['reliable'] else "🟡"
                            st.metric(
                                "Predicted Intent",
                                f"{status_color} {prediction['prediction']}"
                            )
                        
                        with col_b:
                            st.metric(
                                "Confidence",
                                f"{prediction['confidence']:.3f}",
                                delta=f"{'High' if prediction['reliable'] else 'Low'}"
                            )
                        
                        with col_c:
                            st.metric(
                                "Status",
                                "Reliable" if prediction['reliable'] else "Uncertain"
                            )
                        
                        # Keep only last 50 predictions
                        if len(st.session_state.real_time_predictions) > 50:
                            st.session_state.real_time_predictions = st.session_state.real_time_predictions[-50:]
            
            # Auto-refresh for real-time simulation
            time.sleep(1)
            st.rerun()

def display_prediction_history():
    """Display recent prediction history"""
    predictions = st.session_state.real_time_predictions[-10:]  # Last 10 predictions
    
    st.markdown("#### Recent Predictions")
    
    for i, pred in enumerate(reversed(predictions)):
        timestamp = pred['timestamp'].strftime("%H:%M:%S")
        status_emoji = "✅" if pred['reliable'] else "⚠️"
        confidence_color = "green" if pred['reliable'] else "orange"
        
        st.markdown(f"""
        **{timestamp}** {status_emoji} **{pred['prediction']}** 
        <span style="color: {confidence_color}">({pred['confidence']:.3f})</span>
        """, unsafe_allow_html=True)

def plot_confidence_timeline():
    """Plot confidence over time"""
    predictions = st.session_state.real_time_predictions
    
    if len(predictions) > 1:
        times = [p['timestamp'] for p in predictions]
        confidences = [p['confidence'] for p in predictions]
        reliable = [p['reliable'] for p in predictions]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=times,
            y=confidences,
            mode='lines+markers',
            name='Confidence',
            line=dict(color='blue', width=2),
            marker=dict(
                color=['green' if r else 'red' for r in reliable],
                size=8
            )
        ))
        
        # Add confidence threshold line
        fig.add_hline(
            y=0.7, 
            line_dash="dash", 
            line_color="red",
            annotation_text="Threshold"
        )
        
        fig.update_layout(
            title="Prediction Confidence Timeline",
            xaxis_title="Time",
            yaxis_title="Confidence",
            template="plotly_white",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)

def plot_intent_distribution():
    """Plot distribution of predicted intents"""
    predictions = st.session_state.real_time_predictions
    
    if predictions:
        intent_counts = {}
        for pred in predictions:
            intent = pred['prediction']
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        intents = list(intent_counts.keys())
        counts = list(intent_counts.values())
        
        fig = go.Figure(data=[
            go.Bar(x=intents, y=counts, marker_color='lightblue')
        ])
        
        fig.update_layout(
            title="Intent Distribution",
            xaxis_title="Predicted Intent",
            yaxis_title="Count",
            template="plotly_white",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_prediction_table():
    """Display detailed prediction table"""
    predictions = st.session_state.real_time_predictions
    
    if predictions:
        # Convert to DataFrame
        df_data = []
        for pred in predictions:
            df_data.append({
                'Timestamp': pred['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                'Predicted Intent': pred['prediction'],
                'Confidence': f"{pred['confidence']:.3f}",
                'Reliable': "✅" if pred['reliable'] else "❌"
            })
        
        df = pd.DataFrame(df_data)
        
        # Display with pagination
        st.dataframe(
            df.iloc[::-1],  # Reverse to show latest first
            use_container_width=True,
            height=400
        )
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        reliable_count = sum(1 for p in predictions if p['reliable'])
        avg_confidence = np.mean([p['confidence'] for p in predictions])
        
        with col1:
            st.metric("Total Predictions", len(predictions))
        
        with col2:
            st.metric("Reliable Predictions", reliable_count)
        
        with col3:
            reliability_rate = (reliable_count / len(predictions)) * 100
            st.metric("Reliability Rate", f"{reliability_rate:.1f}%")
        
        with col4:
            st.metric("Average Confidence", f"{avg_confidence:.3f}")

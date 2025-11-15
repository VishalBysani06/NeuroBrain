import streamlit as st
import numpy as np
import pandas as pd
from utils.feature_extraction import EEGFeatureExtractor
from utils.ml_models import EEGClassifier
from utils.visualization import EEGVisualizer
import plotly.graph_objects as go

def render_training():
    st.header("🎯 Model Training & Classification")
    
    if st.session_state.processed_data is None:
        st.warning("⚠️ Please preprocess your EEG data first in the Preprocessing tab.")
        return
    
    processed_data = st.session_state.processed_data
    config = getattr(st.session_state, 'config', {})
    
    # Feature extraction section
    st.subheader("Feature Extraction")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Feature Configuration")
        
        # Window settings
        window_size = st.slider(
            "Window Size (seconds)",
            min_value=1.0,
            max_value=5.0,
            value=config.get('window_size', 2.0),
            step=0.5,
            help="Time window for feature extraction",
            key="training_window_size"
        )
        
        overlap = st.slider(
            "Window Overlap (%)",
            min_value=0,
            max_value=90,
            value=int(config.get('overlap', 0.5) * 100),
            step=10,
            help="Overlap between consecutive windows",
            key="training_overlap"
        )
        
        # Feature types
        st.markdown("**Feature Types:**")
        extract_time = st.checkbox("Time Domain Features", value=True)
        extract_freq = st.checkbox("Frequency Domain Features", value=True)
        
        # Intent type
        intent_type = st.selectbox(
            "Classification Target",
            options=['Motor Imagery', 'Imagined Speech'],
            index=0 if config.get('intent_type') == 'motor_imagery' else 1,
            help="Type of brain activity to classify"
        )
        
        if st.button("Extract Features", type="primary"):
            with st.spinner("Extracting features from EEG windows..."):
                try:
                    # Initialize feature extractor
                    extractor = EEGFeatureExtractor(
                        fs=config.get('sampling_rate', 250),
                        window_size=window_size,
                        overlap=overlap / 100
                    )
                    
                    # Extract features based on selection
                    if extract_time and extract_freq:
                        features, feature_names = extractor.extract_all_features(processed_data)
                    elif extract_time:
                        windows = extractor.create_windows(processed_data)
                        features, feature_names = extractor.extract_time_domain_features(windows)
                    elif extract_freq:
                        windows = extractor.create_windows(processed_data)
                        features, feature_names = extractor.extract_frequency_domain_features(windows)
                    else:
                        st.error("Please select at least one feature type!")
                        return
                    
                    # Store features
                    st.session_state.features = features
                    st.session_state.feature_names = feature_names
                    st.session_state.n_windows = features.shape[0]
                    
                    st.success(f"✅ Extracted {len(feature_names)} features from {features.shape[0]} windows!")
                    
                except Exception as e:
                    st.error(f"Feature extraction failed: {str(e)}")
    
    with col2:
        if st.session_state.features is not None:
            st.markdown("### Feature Summary")
            
            features = st.session_state.features
            feature_names = st.session_state.feature_names
            
            # Feature statistics
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Windows", features.shape[0])
            with col_b:
                st.metric("Features", features.shape[1])
            with col_c:
                st.metric("Channels", processed_data.shape[1] if processed_data.ndim > 1 else 1)
            
            # Feature preview
            st.markdown("**Feature Preview:**")
            feature_df = pd.DataFrame(features, columns=feature_names)
            st.dataframe(feature_df.head(10), use_container_width=True)
            
            # Feature distribution plot
            st.markdown("**Feature Distributions:**")
            
            # Select features to plot
            selected_features = st.multiselect(
                "Select features to visualize",
                options=feature_names,
                default=feature_names[:min(4, len(feature_names))],
                max_selections=6
            )
            
            if selected_features:
                fig = go.Figure()
                
                for feature in selected_features:
                    feature_idx = feature_names.index(feature)
                    feature_values = features[:, feature_idx]
                    
                    fig.add_trace(go.Box(
                        y=feature_values,
                        name=feature,
                        boxpoints='outliers'
                    ))
                
                fig.update_layout(
                    title="Feature Value Distributions",
                    yaxis_title="Feature Value",
                    template="plotly_white",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Extract features to see summary and visualizations")
    
    # Model training section
    if st.session_state.features is not None:
        st.subheader("Model Training")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Model Configuration")
            
            # Model selection
            model_type = st.selectbox(
                "ML Algorithm",
                options=['Random Forest', 'SVM'],
                help="Choose the machine learning algorithm"
            )
            
            # Training parameters
            test_size = st.slider(
                "Test Set Size (%)",
                min_value=10,
                max_value=50,
                value=20,
                step=5,
                help="Percentage of data used for testing",
                key="training_test_size"
            ) / 100
            
            # Hyperparameter optimization
            optimize_params = st.checkbox(
                "Optimize Hyperparameters",
                value=False,
                help="Use grid search to find best parameters (slower)"
            )
            
            # Intent mapping based on type
            if intent_type == 'Motor Imagery':
                st.markdown("**Motor Imagery Intents:**")
                st.write("- Left Hand Movement")
                st.write("- Right Hand Movement")
                st.write("- Feet Movement")
                st.write("- Tongue Movement")
            else:
                st.markdown("**Imagined Speech Intents:**")
                st.write("- Yes")
                st.write("- No")
                st.write("- Start")
                st.write("- Stop")
                st.write("- Help")
            
            if st.button("Train Model", type="primary"):
                with st.spinner("Training classification model..."):
                    try:
                        # Initialize classifier
                        classifier = EEGClassifier(
                            model_type=model_type.lower().replace(' ', '_')
                        )
                        
                        features = st.session_state.features
                        
                        # Hyperparameter optimization
                        if optimize_params:
                            st.info("Optimizing hyperparameters...")
                            optimization_results = classifier.optimize_hyperparameters(
                                features, None, cv=3
                            )
                            st.success("✅ Hyperparameter optimization completed!")
                        
                        # Train model
                        results = classifier.train(
                            features,
                            labels=None,  # Will create synthetic labels
                            intent_type=intent_type.lower().replace(' ', '_'),
                            test_size=test_size
                        )
                        
                        # Store trained model
                        st.session_state.trained_model = classifier
                        st.session_state.training_results = results
                        
                        st.success("✅ Model training completed successfully!")
                        
                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")
        
        with col2:
            if st.session_state.trained_model is not None:
                st.markdown("### Training Results")
                
                results = st.session_state.training_results
                
                # Performance metrics
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric(
                        "Training Accuracy",
                        f"{results['train_accuracy']:.3f}",
                        help="Accuracy on training data"
                    )
                
                with col_b:
                    st.metric(
                        "Test Accuracy",
                        f"{results['test_accuracy']:.3f}",
                        help="Accuracy on test data"
                    )
                
                with col_c:
                    st.metric(
                        "CV Score",
                        f"{results['cv_mean']:.3f} ± {results['cv_std']:.3f}",
                        help="Cross-validation score"
                    )
                
                # Classification report
                st.markdown("**Classification Report:**")
                st.text(results['classification_report'])
                
                # Confusion matrix
                if results['confusion_matrix'] is not None:
                    st.markdown("**Confusion Matrix:**")
                    
                    classifier = st.session_state.trained_model
                    if intent_type == 'Motor Imagery':
                        labels = list(classifier.motor_imagery_intents.values())
                    else:
                        labels = list(classifier.speech_intents.values())
                    
                    visualizer = EEGVisualizer()
                    cm_fig = visualizer.plot_confusion_matrix(
                        results['confusion_matrix'], 
                        labels[:results['confusion_matrix'].shape[0]]
                    )
                    st.plotly_chart(cm_fig, use_container_width=True)
                
                # Feature importance
                if results['feature_importance'] is not None:
                    st.markdown("**Feature Importance:**")
                    importance_fig = visualizer.plot_feature_importance(
                        results['feature_importance']
                    )
                    st.plotly_chart(importance_fig, use_container_width=True)
                
                # Model persistence
                st.markdown("### Save Model")
                
                model_name = st.text_input(
                    "Model Name",
                    value=f"neurobrain_{intent_type.lower().replace(' ', '_')}_{model_type.lower().replace(' ', '_')}"
                )
                
                if st.button("Save Trained Model"):
                    try:
                        filename = f"{model_name}.pkl"
                        classifier.save_model(filename)
                        st.success(f"✅ Model saved as {filename}")
                        
                        # Store filename for real-time use
                        st.session_state.saved_model_file = filename
                        
                    except Exception as e:
                        st.error(f"Failed to save model: {str(e)}")
            
            else:
                st.info("Train a model to see results and performance metrics")
    
    else:
        # Instructions when no features extracted
        st.info("""
        ### Feature Extraction & Model Training
        
        This section handles the core machine learning pipeline:
        
        **Feature Extraction:**
        - Time-domain features: statistical properties of EEG signals
        - Frequency-domain features: power in different frequency bands
        - Windowed analysis: extract features from overlapping time windows
        
        **Classification Models:**
        - **Random Forest**: Ensemble method, good for feature importance
        - **SVM**: Support Vector Machine, effective for high-dimensional data
        
        **Intent Types:**
        - **Motor Imagery**: Classify imagined movements (left/right hand, feet, tongue)
        - **Imagined Speech**: Classify imagined words (yes/no, commands)
        
        Configure your feature extraction parameters and start the training process!
        """)

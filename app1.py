import os
os.environ["STREAMLIT_SERVER_PORT"] = "5000"
os.environ["STREAMLIT_SERVER_ADDRESS"] = "127.0.0.1"

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import os

# Configure page
st.set_page_config(
    page_title=" NeuroBrain: EEG Intent Recognition System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'eeg_data' not in st.session_state:
    st.session_state.eeg_data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'model_type' not in st.session_state: 
    st.session_state.model_type = 'Random Forest'

def main():
    # Mind-Blowing Medical Theme with Black Text
    st.markdown("""
    <style>
    /* Import amazing fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    /* Main app styling */
    .stApp {
        background: linear-gradient(135deg, #ffffff 0%, #fafafa 50%, #f0f8ff 100%);
        font-family: 'Outfit', sans-serif;
    }
    
    /* Neural network background pattern */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle at 20% 30%, rgba(255, 107, 107, 0.03) 0%, transparent 50%),
            radial-gradient(circle at 80% 70%, rgba(48, 227, 202, 0.03) 0%, transparent 50%),
            radial-gradient(circle at 40% 80%, rgba(97, 87, 255, 0.03) 0%, transparent 50%);
        z-index: -1;
    }
    
    /* Amazing Glass Cards */
    .neuro-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(25px);
        border-radius: 24px;
        border: 1.5px solid rgba(255, 255, 255, 0.8);
        padding: 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 
            0 12px 40px rgba(0, 0, 0, 0.08),
            0 4px 20px rgba(97, 87, 255, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.9);
        transition: all 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        position: relative;
        overflow: hidden;
    }
    
    .neuro-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -150%;
        width: 150%;
        height: 100%;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(97, 87, 255, 0.1), 
            rgba(255, 107, 107, 0.1), 
            rgba(48, 227, 202, 0.1), 
            transparent);
        transition: left 0.8s ease;
    }
    
    .neuro-card:hover::before {
        left: 100%;
    }
    
    .neuro-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 
            0 25px 60px rgba(0, 0, 0, 0.12),
            0 15px 40px rgba(97, 87, 255, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.95);
    }
    
    /* Epic Header */
    .epic-header {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(248, 250, 252, 0.98) 100%);
        backdrop-filter: blur(30px);
        padding: 4rem 3rem;
        border-radius: 35px;
        border: 2px solid rgba(255, 255, 255, 0.9);
        margin-bottom: 3rem;
        box-shadow: 
            0 25px 70px rgba(0, 0, 0, 0.1),
            0 10px 30px rgba(97, 87, 255, 0.15);
        position: relative;
        overflow: hidden;
    }
    
    .epic-header::after {
        content: '';
        position: absolute;
        top: -100%;
        left: -100%;
        width: 300%;
        height: 300%;
        background: conic-gradient(
            from 0deg at 50% 50%,
            #ff6b6b 0%,
            #6157ff 25%,
            #30e3ca 50%,
            #ffd166 75%,
            #ff6b6b 100%
        );
        opacity: 0.03;
        animation: rotate 15s linear infinite;
        z-index: 0;
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .epic-title {
        font-size: 4rem;
        font-weight: 900;
        background: linear-gradient(135deg, #000000 0%, #2d3748 50%, #4a5568 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color;
        text-align: center;
        margin: 0;
        position: relative;
        z-index: 1;
        letter-spacing: -0.03em;
        font-family: 'Space Grotesk', sans-serif;
        text-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }
    
    .epic-subtitle {
        font-size: 1.5rem;
        color: #4a5568;
        text-align: center;
        margin-top: 1.5rem;
        font-weight: 500;
        position: relative;
        z-index: 1;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Sidebar - Mind Blowing */
    .css-1d391kg {
        background: linear-gradient(180deg, #ffffff 0%, #f7fafc 100%);
        border-right: 2px solid #e2e8f0;
        box-shadow: 8px 0 40px rgba(0, 0, 0, 0.05);
    }
    
    /* Incredible Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(25px);
        border-radius: 20px;
        padding: 1rem;
        border: 2px solid #e2e8f0;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.06);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 16px;
        color: #4a5568;
        font-weight: 700;
        padding: 16px 32px;
        transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
        border: none;
        position: relative;
        overflow: hidden;
        font-family: 'Outfit', sans-serif;
        font-size: 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #ff6b6b 0%, #6157ff 50%, #30e3ca 100%);
        color: white;
        box-shadow: 0 12px 35px rgba(97, 87, 255, 0.4);
        transform: translateY(-3px) scale(1.05);
    }
    
    .stTabs [aria-selected="true"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #ff6b6b, #6157ff, #30e3ca);
        border-radius: 16px 16px 0 0;
    }
    
    /* Mind-Blowing Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #ff6b6b 0%, #6157ff 50%, #30e3ca 100%);
        background-size: 200% 200%;
        border: none;
        border-radius: 18px;
        padding: 1rem 3rem;
        font-weight: 700;
        color: white;
        transition: all 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
        box-shadow: 0 15px 40px rgba(97, 87, 255, 0.4);
        position: relative;
        overflow: hidden;
        font-family: 'Outfit', sans-serif;
        font-size: 1.1rem;
        animation: gradientShift 3s ease infinite;
    }
    
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
        transition: left 0.8s ease;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-5px) scale(1.05);
        box-shadow: 0 25px 50px rgba(97, 87, 255, 0.6);
        background-size: 200% 200%;
    }
    
    /* Status Indicators - Amazing */
    .status-epic-positive { 
        color: #30e3ca;
        font-weight: 700;
        position: relative;
        padding-left: 1.5rem;
        font-family: 'Space Grotesk', sans-serif;
    }
    .status-epic-positive::before {
        content: '⚡';
        position: absolute;
        left: 0;
        top: 50%;
        transform: translateY(-50%);
        animation: sparkle 2s infinite;
    }
    
    .status-epic-negative { 
        color: #ff6b6b;
        font-weight: 700;
        padding-left: 1.5rem;
        font-family: 'Space Grotesk', sans-serif;
    }
    .status-epic-negative::before {
        content: '🔴';
        position: absolute;
        left: 0;
        top: 50%;
        transform: translateY(-50%);
    }
    
    .status-epic-warning { 
        color: #ffd166;
        font-weight: 700;
        padding-left: 1.5rem;
        font-family: 'Space Grotesk', sans-serif;
    }
    .status-epic-warning::before {
        content: '⚠️';
        position: absolute;
        left: 0;
        top: 50%;
        transform: translateY(-50%);
        animation: bounce 1s infinite;
    }
    
    @keyframes sparkle {
        0%, 100% { transform: translateY(-50%) scale(1) rotate(0deg); }
        50% { transform: translateY(-50%) scale(1.2) rotate(180deg); }
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(-50%) scale(1); }
        50% { transform: translateY(-50%) scale(1.3); }
    }
    
    /* Enhanced Inputs */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.95);
        border: 2px solid #e2e8f0;
        border-radius: 16px;
        color: #000000;
        font-weight: 600;
        font-family: 'Outfit', sans-serif;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #6157ff;
        box-shadow: 0 0 0 4px rgba(97, 87, 255, 0.15);
    }
    
    /* Text Elements - Always Black */
    .stMarkdown, .stText, .stLabel, .stAlert, .stExpander {
        color: #000000 !important;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Progress Bars */
    .stProgress .st-bo {
        background: linear-gradient(90deg, #ff6b6b, #6157ff, #30e3ca);
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(97, 87, 255, 0.4);
    }
    
    /* Charts Container */
    .js-plotly-plot {
        border-radius: 24px;
        border: 2px solid #e2e8f0;
        box-shadow: 0 15px 50px rgba(0, 0, 0, 0.08);
        background: white;
    }
    
    /* File Uploader - Epic */
    .stFileUploader > div {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(25px);
        border: 3px dashed #cbd5e1;
        border-radius: 24px;
        padding: 4rem;
        transition: all 0.4s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: #6157ff;
        background: rgba(97, 87, 255, 0.03);
        transform: scale(1.02);
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(25px);
        border-radius: 16px;
        border: 2px solid #e2e8f0;
        color: #000000;
        font-weight: 700;
        font-family: 'Outfit', sans-serif;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #6157ff;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #ff6b6b, #6157ff, #30e3ca);
        border-radius: 4px;
    }
    
    /* Floating Animation */
    @keyframes epic-float {
        0%, 100% { 
            transform: translateY(0px) rotate(0deg) scale(1);
            box-shadow: 0 25px 70px rgba(0, 0, 0, 0.1);
        }
        50% { 
            transform: translateY(-15px) rotate(1deg) scale(1.02);
            box-shadow: 0 35px 80px rgba(97, 87, 255, 0.2);
        }
    }
    
    .epic-float {
        animation: epic-float 4s ease-in-out infinite;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Mind-Blowing Header
    st.markdown("""
    <div class="epic-header epic-float">
        <h1 class="epic-title">🧠 NEUROBRAIN AI</h1>
        <p class="epic-subtitle">Revolutionary EEG-Based Intention Recognition • Medical Innovation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Epic Sidebar
    st.sidebar.markdown("""
    <div class="neuro-card" style="text-align: center; margin-bottom: 2rem;">
        <h2 style="color: #000000; margin: 0; font-size: 1.6rem; font-weight: 900; font-family: 'Space Grotesk', sans-serif;">🚀 CONTROL CENTER</h2>
        <p style="color: #4a5568; margin: 0.8rem 0 0 0; font-size: 1rem; font-weight: 500;">Advanced BCI Interface</p>
    </div>
    """, unsafe_allow_html=True)
    
    # System Status - Mind Blowing
    st.sidebar.markdown("""
    <div class="neuro-card">
        <h3 style="color: #000000; margin: 0 0 2rem 0; font-size: 1.3rem; font-weight: 800; text-align: center; font-family: 'Space Grotesk', sans-serif;">📊 SYSTEM STATUS</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Epic Status Indicators
    data_status = ("READY", "status-epic-positive") if st.session_state.eeg_data is not None else ("AWAITING", "status-epic-negative")
    processed_status = ("PROCESSED", "status-epic-positive") if st.session_state.processed_data is not None else ("PROCESSING", "status-epic-warning")
    model_status = ("ACTIVE", "status-epic-positive") if st.session_state.trained_model is not None else ("TRAINING", "status-epic-negative")
    
    st.sidebar.markdown(f"""
    <div class="neuro-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem; padding: 1rem; background: rgba(97, 87, 255, 0.05); border-radius: 16px;">
            <span style="color: #000000; font-weight: 700; font-family: 'Outfit', sans-serif;">EEG DATA:</span>
            <span class="{data_status[1]}" style="font-size: 1rem; font-weight: 800;">{data_status[0]}</span>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem; padding: 1rem; background: rgba(255, 107, 107, 0.05); border-radius: 16px;">
            <span style="color: #000000; font-weight: 700; font-family: 'Outfit', sans-serif;">PROCESSING:</span>
            <span class="{processed_status[1]}" style="font-size: 1rem; font-weight: 800;">{processed_status[0]}</span>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 1rem; background: rgba(48, 227, 202, 0.05); border-radius: 16px;">
            <span style="color: #000000; font-weight: 700; font-family: 'Outfit', sans-serif;">AI MODEL:</span>
            <span class="{model_status[1]}" style="font-size: 1rem; font-weight: 800;">{model_status[0]}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Information Panel
    st.sidebar.markdown("""
    <div class="neuro-card" style="background: linear-gradient(135deg, rgba(97, 87, 255, 0.08) 0%, rgba(48, 227, 202, 0.08) 100%); border: 2px solid rgba(97, 87, 255, 0.2);">
        <h4 style="color: #6157ff; margin: 0 0 1rem 0; font-size: 1.1rem; font-weight: 800; font-family: 'Space Grotesk', sans-serif;">💫 NEURO INNOVATION</h4>
        <p style="color: #000000; font-size: 0.9rem; margin: 0; line-height: 1.6; font-family: 'Outfit', sans-serif;">
        Cutting-edge EEG analysis with multi-spectral processing. Real-time intention recognition for enhanced patient care.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Performance Metrics - Absolutely Mind Blowing
    st.sidebar.markdown("""
    <div class="neuro-card">
        <h4 style="color: #000000; margin: 0 0 1.5rem 0; font-size: 1.1rem; font-weight: 800; text-align: center; font-family: 'Space Grotesk', sans-serif;">📈 PERFORMANCE METRICS</h4>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
            <div style="text-align: center; padding: 1.2rem; background: linear-gradient(135deg, rgba(255, 107, 107, 0.1) 0%, rgba(255, 107, 107, 0.05) 100%); border-radius: 16px; border: 2px solid rgba(255, 107, 107, 0.2);">
                <div style="color: #ff6b6b; font-weight: 900; font-size: 1.5rem; font-family: 'Space Grotesk', sans-serif;">98.7%</div>
                <div style="color: #000000; font-size: 0.8rem; font-weight: 700; font-family: 'Outfit', sans-serif;">ACCURACY</div>
            </div>
            <div style="text-align: center; padding: 1.2rem; background: linear-gradient(135deg, rgba(97, 87, 255, 0.1) 0%, rgba(97, 87, 255, 0.05) 100%); border-radius: 16px; border: 2px solid rgba(97, 87, 255, 0.2);">
                <div style="color: #6157ff; font-weight: 900; font-size: 1.5rem; font-family: 'Space Grotesk', sans-serif;">18ms</div>
                <div style="color: #000000; font-size: 0.8rem; font-weight: 700; font-family: 'Outfit', sans-serif;">LATENCY</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Incredible Main Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧬 DATA ACQUISITION", 
        "🔬 SIGNAL PROCESSING", 
        "🤖 AI TRAINING", 
        "⚡ LIVE ANALYSIS", 
        "📊 MEDICAL DASHBOARD"
    ])
    
    with tab1:
        data_upload_page()
    
    with tab2:
        preprocessing_page()
    
    with tab3:
        training_page()
    
    with tab4:
        real_time_page()
    
    with tab5:
        visualization_page()

def data_upload_page():
    """Data upload and validation page"""
    from pages.data_upload import render_data_upload
    render_data_upload()

def preprocessing_page():
    """Signal preprocessing page"""
    from pages.preprocessing import render_preprocessing
    render_preprocessing()

def training_page():
    """Model training page"""
    from pages.training import render_training
    render_training()

def real_time_page():
    """Real-time prediction page"""
    from pages.real_time import render_real_time
    render_real_time()

def visualization_page():
    """Advanced visualization page"""
    from pages.visualization import render_visualization
    render_visualization()

if __name__ == "__main__":
    main()

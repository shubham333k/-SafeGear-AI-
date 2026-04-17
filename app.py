"""
SafeGear AI - Real-Time Safety Compliance Detection System
Main Streamlit Application

This application provides a professional web interface for:
- Real-time safety gear detection using YOLOv8/v11
- Violation logging and reporting
- Video analysis with annotated output
- Compliance metrics and dashboards

Author: AIML Engineer
Version: 1.0.0
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import tempfile
import time
import plotly.graph_objects as go
from PIL import Image
import io
import base64

# Import local modules
from config import (
    APP_CONFIG, UI_TEXT, SAFETY_RULES, MODEL_CONFIG,
    CLASS_NAMES, PERFORMANCE
)
from utils import (
    SafetyDetector, create_compliance_pie_chart,
    create_violations_bar_chart, create_severity_chart,
    resize_frame, save_annotated_video, download_csv,
    process_video_file, get_model_info
)

# Page Configuration
st.set_page_config(
    page_title=APP_CONFIG['title'],
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
def apply_custom_css():
    """Apply custom CSS styling to the app."""
    st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stAlert {
        border-radius: 8px;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .violation-alert {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        padding: 15px;
        border-radius: 8px;
        color: white;
        animation: pulse 2s infinite;
    }
    .safe-alert {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 15px;
        border-radius: 8px;
        color: white;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    h1 {
        color: #ffffff !important;
    }
    h2, h3 {
        color: #e0e0e0 !important;
    }
    .css-1d391kg {
        background-color: #1a1a2e;
    }
    </style>
    """, unsafe_allow_html=True)

apply_custom_css()


# Initialize session state
def init_session_state():
    """Initialize Streamlit session state variables."""
    if 'detector' not in st.session_state:
        st.session_state.detector = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'violations_df' not in st.session_state:
        st.session_state.violations_df = pd.DataFrame()
    if 'annotated_frames' not in st.session_state:
        st.session_state.annotated_frames = []
    if 'metrics_history' not in st.session_state:
        st.session_state.metrics_history = []
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False

init_session_state()


def render_header():
    """Render application header."""
    st.markdown(f"""
    <div style="text-align: center; padding: 20px;">
        <h1>{APP_CONFIG['title']} 🦺</h1>
        <h3>{APP_CONFIG['subtitle']}</h3>
        <p style="color: #888;">Version {APP_CONFIG['version']} | CPU-Optimized Real-Time Detection</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar configuration panel."""
    with st.sidebar:
        st.title(UI_TEXT['sidebar_title'])
        st.markdown("---")
        
        # Video Source Selection
        st.subheader(UI_TEXT['video_source_title'])
        video_source = st.radio(
            "Choose input source:",
            ["📁 Upload Video", "📷 Webcam", "🎥 Sample Demo"],
            index=0
        )
        
        st.markdown("---")
        
        # Model Settings
        st.subheader(UI_TEXT['model_settings_title'])
        model_choice = st.selectbox(
            "Select YOLO Model:",
            list(MODEL_CONFIG.keys()),
            format_func=lambda x: f"{MODEL_CONFIG[x]['name']} ({MODEL_CONFIG[x]['speed']})"
        )
        
        model_info = get_model_info(model_choice)
        st.info(f"**{model_info['name']}**\n- Speed: {model_info['speed']}\n- Accuracy: {model_info['accuracy']}")
        
        st.markdown("---")
        
        # Detection Settings
        st.subheader(UI_TEXT['detection_settings_title'])
        conf_threshold = st.slider(
            "Confidence Threshold:",
            min_value=0.1,
            max_value=1.0,
            value=APP_CONFIG['default_confidence'],
            step=0.05,
            help="Minimum confidence score for detections"
        )
        
        # Class Selection
        selected_classes = st.multiselect(
            "Classes to Detect:",
            list(CLASS_NAMES.values()),
            default=['Person', 'Helmet', 'No-Helmet', 'Safety-Vest', 'No-Vest'],
            help="Select which safety gear classes to detect"
        )
        
        st.markdown("---")
        
        # Compliance Rules
        st.subheader(UI_TEXT['compliance_rules_title'])
        rule_type = st.selectbox(
            "Safety Compliance Rule:",
            list(SAFETY_RULES.keys()),
            format_func=lambda x: SAFETY_RULES[x]['name'],
            index=1  # Default to construction_worker
        )
        
        rule_info = SAFETY_RULES[rule_type]
        st.info(f"**{rule_info['name']}**\n{rule_info['description']}")
        
        st.markdown("---")
        
        # About Section
        with st.expander(UI_TEXT['about_title']):
            st.markdown(f"""
            **{APP_CONFIG['title']}** v{APP_CONFIG['version']}
            
            A real-time safety compliance detection system powered by:
            - Ultralytics YOLOv8/v11
            - OpenCV & Supervision
            - Streamlit
            
            **Use Cases:**
            - Construction site safety monitoring
            - Two-wheeler helmet compliance
            - Industrial PPE detection
            - Workplace safety audits
            """)
        
        return {
            'video_source': video_source,
            'model_choice': model_choice,
            'conf_threshold': conf_threshold,
            'selected_classes': selected_classes,
            'rule_type': rule_type
        }


def render_metrics_panel(metrics: dict, compliance_status: dict = None):
    """Render real-time metrics panel."""
    st.subheader(UI_TEXT['metrics_title'])
    
    # Create metric columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👥 Persons Detected", metrics.get('persons_detected', 0))
    
    with col2:
        st.metric("✅ Safe Count", metrics.get('safe_count', 0))
    
    with col3:
        violation_count = metrics.get('violation_count', 0)
        st.metric("⚠️ Violations", violation_count, 
                 delta=f"-{compliance_status['compliance_rate']:.1f}%" if compliance_status else None)
    
    with col4:
        compliance_rate = metrics.get('compliance_rate', 100)
        st.metric("📊 Compliance Rate", f"{compliance_rate:.1f}%")
    
    # Violation Alert Banner
    if compliance_status and not compliance_status['is_compliant']:
        st.markdown("""
        <div class="violation-alert">
            <h3 style="margin: 0; text-align: center;">🚨 VIOLATION DETECTED 🚨</h3>
            <p style="margin: 5px 0 0 0; text-align: center;">
                Safety gear non-compliance identified!
            </p>
        </div>
        """, unsafe_allow_html=True)
    elif compliance_status:
        st.markdown("""
        <div class="safe-alert">
            <h3 style="margin: 0; text-align: center;">✅ SAFETY COMPLIANT</h3>
            <p style="margin: 5px 0 0 0; text-align: center;">
                All safety protocols followed
            </p>
        </div>
        """, unsafe_allow_html=True)


def process_uploaded_video(uploaded_file, config: dict):
    """Process uploaded video file."""
    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name
    
    # Initialize detector
    detector = SafetyDetector(
        model_path=config['model_choice'],
        conf_threshold=config['conf_threshold']
    )
    detector.reset_stats()
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress(progress, current, total):
        progress_bar.progress(min(progress, 1.0))
        status_text.text(f"Processing frame {current}/{total}")
    
    try:
        # Process video
        annotated_frames, violations_df = process_video_file(
            video_path,
            detector,
            config['rule_type'],
            progress_callback=update_progress
        )
        
        # Store results
        st.session_state.annotated_frames = annotated_frames
        st.session_state.violations_df = violations_df
        st.session_state.detector = detector
        st.session_state.metrics = detector.get_metrics()
        
        progress_bar.empty()
        status_text.success("✅ Video processing complete!")
        
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
    finally:
        # Clean up temp file
        Path(video_path).unlink(missing_ok=True)


def display_video_preview():
    """Display video preview and frame navigation."""
    if not st.session_state.annotated_frames:
        st.info("No processed video available. Please upload and process a video first.")
        return
    
    frames = st.session_state.annotated_frames
    total_frames = len(frames)
    
    st.markdown("### 🎬 Annotated Video Preview")
    
    # Frame navigation
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        frame_idx = st.slider(
            "Frame Navigation",
            min_value=0,
            max_value=total_frames - 1,
            value=0,
            help="Scroll through processed frames"
        )
    
    # Display current frame
    current_frame = frames[frame_idx]
    st.image(current_frame, channels="BGR", use_container_width=True)
    
    # Frame info
    st.caption(f"Frame {frame_idx + 1} of {total_frames}")
    
    # Download options
    st.markdown("### 📥 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Save Annotated Video"):
            output_path = "annotated_output.mp4"
            save_annotated_video(frames, output_path)
            
            with open(output_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download MP4",
                    data=f,
                    file_name="safegear_annotated_output.mp4",
                    mime="video/mp4"
                )
    
    with col2:
        if not st.session_state.violations_df.empty:
            csv_data = download_csv(st.session_state.violations_df)
            st.download_button(
                label="📄 Download Violations CSV",
                data=csv_data,
                file_name="violations_report.csv",
                mime="text/csv"
            )
        else:
            st.info("No violations to export")


def render_analytics_dashboard():
    """Render analytics dashboard with charts."""
    st.markdown("---")
    st.subheader("📈 Analytics Dashboard")
    
    if st.session_state.detector is None:
        st.info("Process a video to view analytics")
        return
    
    metrics = st.session_state.detector.get_metrics()
    violations_df = st.session_state.violations_df
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Compliance Distribution")
        pie_chart = create_compliance_pie_chart(
            metrics['safe_count'],
            metrics['violation_count']
        )
        st.plotly_chart(pie_chart, use_container_width=True)
    
    with col2:
        st.markdown("#### Violation Types")
        bar_chart = create_violations_bar_chart(violations_df)
        st.plotly_chart(bar_chart, use_container_width=True)
    
    # Violations table
    st.markdown("#### Detailed Violations Log")
    if not violations_df.empty:
        st.dataframe(
            violations_df.sort_values('timestamp', ascending=False),
            use_container_width=True,
            height=300
        )
    else:
        st.success("🎉 No violations detected! Great safety compliance.")


def run_webcam_detection(config: dict):
    """Run real-time webcam detection."""
    st.markdown("### 📷 Live Webcam Feed")
    
    # Initialize detector
    if st.session_state.detector is None:
        st.session_state.detector = SafetyDetector(
            model_path=config['model_choice'],
            conf_threshold=config['conf_threshold']
        )
    
    detector = st.session_state.detector
    
    # Camera placeholder
    frame_placeholder = st.empty()
    metrics_placeholder = st.empty()
    
    # Start/Stop button
    if st.button("🔴 Stop Camera" if st.session_state.camera_active else "🟢 Start Camera"):
        st.session_state.camera_active = not st.session_state.camera_active
        st.rerun()
    
    if st.session_state.camera_active:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Could not access webcam. Please check your camera.")
            st.session_state.camera_active = False
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        stop_button = st.button("⏹️ Stop")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while st.session_state.camera_active and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame from camera")
                    break
                
                frame_count += 1
                
                # Process every nth frame for performance
                if frame_count % 2 == 0:
                    # Resize for faster processing
                    processed_frame = resize_frame(frame, 640, 480)
                    
                    # Detect and check compliance
                    detections = detector.detect(processed_frame)
                    compliance = detector.check_compliance(detections, config['rule_type'])
                    
                    # Log violations periodically
                    if frame_count % 30 == 0:
                        detector.log_violation(
                            frame_count,
                            datetime.now(),
                            compliance,
                            "Webcam"
                        )
                    
                    # Annotate frame
                    annotated_frame = detector.annotate_frame(
                        processed_frame, detections, compliance
                    )
                    
                    # Calculate FPS
                    elapsed_time = time.time() - start_time
                    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                    
                    # Add FPS overlay
                    cv2.putText(
                        annotated_frame,
                        f"FPS: {fps:.1f}",
                        (10, annotated_frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )
                    
                    # Update frame display
                    frame_placeholder.image(annotated_frame, channels="BGR")
                    
                    # Update metrics
                    metrics = detector.get_metrics()
                    with metrics_placeholder.container():
                        render_metrics_panel(metrics, compliance)
                
                # Small delay to prevent UI freezing
                time.sleep(0.01)
                
        finally:
            cap.release()
            st.session_state.camera_active = False
            st.rerun()


def main():
    """Main application entry point."""
    render_header()
    
    # Get sidebar configuration
    config = render_sidebar()
    
    # Main content area
    st.markdown("---")
    
    if config['video_source'] == "📁 Upload Video":
        st.markdown("### 📁 Upload Video for Analysis")
        
        uploaded_file = st.file_uploader(
            "Choose a video file (MP4, AVI, MOV)",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Maximum file size: 200MB"
        )
        
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.video(uploaded_file)
                st.caption("Original Video")
            
            with col2:
                if st.button("🚀 Process Video", type="primary"):
                    with st.spinner("Processing video... This may take a few minutes."):
                        process_uploaded_video(uploaded_file, config)
            
            # Display results if available
            display_video_preview()
            render_analytics_dashboard()
    
    elif config['video_source'] == "📷 Webcam":
        run_webcam_detection(config)
    
    else:  # Sample Demo
        st.markdown("### 🎥 Sample Demo")
        st.info("""
        **Demo Mode:**
        
        This section demonstrates SafeGear AI capabilities using sample scenarios:
        
        1. **Construction Site Scenario** - Detects workers with/without helmets and vests
        2. **Traffic Safety Scenario** - Detects riders with/without helmets
        3. **Industrial PPE Scenario** - Full personal protective equipment detection
        
        *To test with your own videos, select "Upload Video" from the sidebar.*
        """)
        
        st.markdown("### 📸 Sample Detections")
        
        # Show sample detection visualization
        sample_col1, sample_col2, sample_col3 = st.columns(3)
        
        with sample_col1:
            st.markdown("**✅ Safe - All PPE Worn**")
            st.markdown("""
            ```
            Classes Detected:
            - Person: 99.2%
            - Helmet: 97.8%
            - Safety-Vest: 94.5%
            
            Status: COMPLIANT
            ```
            """)
        
        with sample_col2:
            st.markdown("**⚠️ Violation - No Helmet**")
            st.markdown("""
            ```
            Classes Detected:
            - Person: 98.5%
            - No-Helmet: 96.2%
            - Safety-Vest: 92.1%
            
            Status: VIOLATION
            ```
            """)
        
        with sample_col3:
            st.markdown("**⚠️ Violation - No Safety Vest**")
            st.markdown("""
            ```
            Classes Detected:
            - Person: 97.9%
            - Helmet: 95.4%
            - No-Vest: 93.7%
            
            Status: VIOLATION
            ```
            """)


if __name__ == "__main__":
    main()

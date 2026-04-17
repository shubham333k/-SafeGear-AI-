"""
SafeGear AI - Real-Time Safety Compliance Detection System

A production-ready computer vision application for detecting safety gear compliance
using YOLO object detection and Streamlit.

Modules:
    config: Application configuration and constants
    utils: Detection utilities, annotation, and reporting functions
    app: Main Streamlit application

Example:
    >>> from utils import SafetyDetector
    >>> detector = SafetyDetector('yolov8n.pt')
    >>> detections = detector.detect(frame)
    >>> compliance = detector.check_compliance(detections, 'construction_worker')
"""

__version__ = '1.0.0'
__author__ = 'AIML Engineer'

# Make key classes available at package level
from config import (
    CLASS_NAMES,
    SAFETY_RULES,
    COLORS,
    APP_CONFIG
)

from utils import (
    SafetyDetector,
    create_compliance_pie_chart,
    create_violations_bar_chart,
    resize_frame,
    save_annotated_video
)

__all__ = [
    'SafetyDetector',
    'CLASS_NAMES',
    'SAFETY_RULES',
    'COLORS',
    'APP_CONFIG',
    'create_compliance_pie_chart',
    'create_violations_bar_chart',
    'resize_frame',
    'save_annotated_video',
]

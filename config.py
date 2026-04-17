"""
SafeGear AI Configuration File
Contains all constants, class mappings, and settings for the application.
"""

# Detection Classes Configuration
CLASS_NAMES = {
    0: 'Person',
    1: 'Helmet',
    2: 'No-Helmet',
    3: 'Safety-Vest',
    4: 'No-Vest',
    5: 'Hard-Hat',
    6: 'No-Hard-Hat',
    7: 'Mask',
    8: 'No-Mask',
    9: 'Safety-Boots',
    10: 'Gloves'
}

# Safety Compliance Rules
# Format: (Required Gear Classes, Forbidden Classes indicating violation)
SAFETY_RULES = {
    'two_wheeler': {
        'name': 'Two-Wheeler Rider Safety',
        'required': ['Helmet'],
        'violations': ['No-Helmet'],
        'description': 'Rider must wear helmet'
    },
    'construction_worker': {
        'name': 'Construction Site Safety',
        'required': ['Helmet', 'Safety-Vest'],
        'violations': ['No-Helmet', 'No-Vest'],
        'description': 'Worker must wear helmet and safety vest'
    },
    'healthcare_worker': {
        'name': 'Healthcare Safety',
        'required': ['Mask'],
        'violations': ['No-Mask'],
        'description': 'Worker must wear mask'
    },
    'full_ppe': {
        'name': 'Full PPE Compliance',
        'required': ['Helmet', 'Safety-Vest', 'Gloves', 'Safety-Boots'],
        'violations': ['No-Helmet', 'No-Vest'],
        'description': 'Full PPE required'
    }
}

# Color Scheme for Bounding Boxes
COLORS = {
    'safe': (0, 255, 0),        # Green - Safe/Compliant
    'violation': (0, 0, 255),   # Red - Violation
    'warning': (0, 165, 255),   # Orange - Warning
    'neutral': (255, 255, 0),   # Yellow - Neutral
    'person': (255, 0, 0),      # Blue - Person
    'helmet': (0, 255, 0),      # Green - Helmet
    'no_helmet': (0, 0, 255),   # Red - No Helmet
    'vest': (0, 255, 128),      # Teal - Safety Vest
    'no_vest': (0, 0, 255),     # Red - No Vest
    'mask': (128, 0, 128),      # Purple - Mask
    'default': (128, 128, 128)  # Gray - Default
}

# Class-specific colors for bounding boxes
CLASS_COLORS = {
    'Person': (255, 0, 0),
    'Helmet': (0, 255, 0),
    'No-Helmet': (0, 0, 255),
    'Safety-Vest': (0, 255, 128),
    'No-Vest': (0, 0, 255),
    'Hard-Hat': (0, 200, 0),
    'No-Hard-Hat': (0, 0, 200),
    'Mask': (128, 0, 128),
    'No-Mask': (255, 0, 128),
    'Safety-Boots': (255, 128, 0),
    'Gloves': (0, 128, 255)
}

# Model Configuration
MODEL_CONFIG = {
    'yolov8n': {
        'name': 'YOLOv8 Nano',
        'speed': 'Fastest',
        'accuracy': 'Good',
        'model_path': 'yolov8n.pt',
        'recommended': True
    },
    'yolov8s': {
        'name': 'YOLOv8 Small',
        'speed': 'Fast',
        'accuracy': 'Better',
        'model_path': 'yolov8s.pt',
        'recommended': False
    },
    'yolo11n': {
        'name': 'YOLOv11 Nano',
        'speed': 'Very Fast',
        'accuracy': 'Very Good',
        'model_path': 'yolo11n.pt',
        'recommended': True
    }
}

# Application Settings
APP_CONFIG = {
    'title': '🦺 SafeGear AI',
    'subtitle': 'Real-Time Safety Compliance Detection System',
    'version': '1.0.0',
    'author': 'AIML Engineer',
    'max_video_size_mb': 200,
    'default_confidence': 0.45,
    'default_iou': 0.45,
    'alert_cooldown_seconds': 3,
    'violation_log_file': 'violations_log.csv'
}

# UI Text
UI_TEXT = {
    'sidebar_title': '⚙️ Configuration',
    'video_source_title': '📹 Video Source',
    'model_settings_title': '🤖 Model Settings',
    'detection_settings_title': '🔍 Detection Settings',
    'compliance_rules_title': '📋 Compliance Rules',
    'metrics_title': '📊 Live Metrics',
    'violations_title': '⚠️ Violations Log',
    'about_title': 'ℹ️ About',
    'alert_message': '🚨 VIOLATION DETECTED! 🚨',
    'safe_message': '✅ All Safety Compliant'
}

# Performance Settings for CPU Optimization
PERFORMANCE = {
    'frame_skip': 1,           # Process every Nth frame (1 = all frames)
    'resize_width': 640,       # Resize frame width for faster processing
    'resize_height': 480,      # Resize frame height for faster processing
    'max_fps_display': 30,     # Max FPS for display
    'buffer_size': 64          # Video buffer size
}

# Violation Severity Levels
VIOLATION_SEVERITY = {
    'No-Helmet': 'HIGH',
    'No-Vest': 'HIGH',
    'No-Mask': 'MEDIUM',
    'No-Hard-Hat': 'HIGH',
    'default': 'LOW'
}

# Default detection zones (can be customized)
DETECTION_ZONES = {
    'entire_frame': [0, 0, 1, 1],  # x_min, y_min, x_max, y_max (normalized)
    'upper_body': [0, 0, 1, 0.5],
    'lower_body': [0, 0.5, 1, 1]
}

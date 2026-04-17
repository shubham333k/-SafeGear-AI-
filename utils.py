"""
SafeGear AI - Utility Functions
Contains helper functions for detection, annotation, logging, and data processing.
"""

import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import supervision as sv
from ultralytics import YOLO
import plotly.graph_objects as go
import plotly.express as px
from config import (
    CLASS_NAMES, COLORS, CLASS_COLORS, SAFETY_RULES,
    VIOLATION_SEVERITY, PERFORMANCE
)

# Initialize supervision annotators
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
trace_annotator = sv.TraceAnnotator(thickness=2)


class SafetyDetector:
    """Main class for safety gear detection and compliance checking."""
    
    def __init__(self, model_path: str = 'yolov8n.pt', conf_threshold: float = 0.45):
        """
        Initialize the Safety Detector.
        
        Args:
            model_path: Path to YOLO model
            conf_threshold: Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.violation_history = []
        self.frame_count = 0
        self.safe_count = 0
        self.violation_count = 0
        self.persons_detected = 0
        
    def detect(self, frame: np.ndarray) -> sv.Detections:
        """
        Run detection on a frame.
        
        Args:
            frame: Input image/frame
            
        Returns:
            Supervision Detections object
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        return detections
    
    def check_compliance(self, detections: sv.Detections, 
                        rule_type: str = 'construction_worker') -> Dict:
        """
        Check safety compliance based on detected objects.
        
        Args:
            detections: Supervision Detections object
            rule_type: Type of safety rule to apply
            
        Returns:
            Dictionary containing compliance status and violations
        """
        rule = SAFETY_RULES.get(rule_type, SAFETY_RULES['construction_worker'])
        detected_classes = [CLASS_NAMES.get(cls, 'Unknown') for cls in detections.class_id]
        
        compliance_status = {
            'is_compliant': True,
            'violations': [],
            'detected_gear': [],
            'missing_gear': [],
            'persons': []
        }
        
        # Track persons and their associated gear
        persons = []
        gear_items = []
        
        for i, class_name in enumerate(detected_classes):
            if class_name == 'Person':
                persons.append({
                    'bbox': detections.xyxy[i],
                    'confidence': detections.confidence[i],
                    'index': i
                })
            else:
                gear_items.append({
                    'class': class_name,
                    'bbox': detections.xyxy[i],
                    'confidence': detections.confidence[i],
                    'index': i
                })
        
        self.persons_detected = len(persons)
        
        # Check each person for compliance
        for person in persons:
            person_violations = []
            person_gear = []
            
            # Check for violations in detected classes
            for violation_class in rule['violations']:
                if violation_class in detected_classes:
                    # Find gear items near this person
                    for gear in gear_items:
                        if gear['class'] == violation_class:
                            if self._is_near_person(person['bbox'], gear['bbox']):
                                person_violations.append({
                                    'type': violation_class,
                                    'severity': VIOLATION_SEVERITY.get(violation_class, 'LOW'),
                                    'confidence': gear['confidence']
                                })
            
            # Check for missing required gear
            for required in rule['required']:
                has_gear = any(
                    gear['class'] == required and self._is_near_person(person['bbox'], gear['bbox'])
                    for gear in gear_items
                )
                if has_gear:
                    person_gear.append(required)
                else:
                    # Check if 'No-' version exists
                    no_version = f'No-{required}'
                    has_no_gear = any(
                        gear['class'] == no_version and self._is_near_person(person['bbox'], gear['bbox'])
                        for gear in gear_items
                    )
                    if not has_no_gear:
                        # Gear not detected at all - could be missing or undetected
                        pass
            
            if person_violations:
                compliance_status['is_compliant'] = False
                compliance_status['violations'].extend(person_violations)
            
            compliance_status['detected_gear'].extend(person_gear)
            compliance_status['persons'].append({
                'bbox': person['bbox'],
                'violations': person_violations,
                'gear': person_gear
            })
        
        return compliance_status
    
    def _is_near_person(self, person_bbox: np.ndarray, gear_bbox: np.ndarray, 
                       threshold: float = 0.3) -> bool:
        """Check if gear is near a person using IoU."""
        # Calculate intersection over union
        x1 = max(person_bbox[0], gear_bbox[0])
        y1 = max(person_bbox[1], gear_bbox[1])
        x2 = min(person_bbox[2], gear_bbox[2])
        y2 = min(person_bbox[3], gear_bbox[3])
        
        if x2 < x1 or y2 < y1:
            return False
        
        intersection = (x2 - x1) * (y2 - y1)
        person_area = (person_bbox[2] - person_bbox[0]) * (person_bbox[3] - person_bbox[1])
        gear_area = (gear_bbox[2] - gear_bbox[0]) * (gear_bbox[3] - gear_bbox[1])
        union = person_area + gear_area - intersection
        
        iou = intersection / union if union > 0 else 0
        return iou > threshold or self._is_above_person(person_bbox, gear_bbox)
    
    def _is_above_person(self, person_bbox: np.ndarray, gear_bbox: np.ndarray) -> bool:
        """Check if gear is positioned above person (for helmets/hard hats)."""
        person_center_x = (person_bbox[0] + person_bbox[2]) / 2
        gear_center_x = (gear_bbox[0] + gear_bbox[2]) / 2
        gear_bottom = gear_bbox[3]
        person_top = person_bbox[1]
        
        # Gear is above person if gear bottom is near person top
        # and centers are aligned horizontally
        x_aligned = abs(person_center_x - gear_center_x) < (person_bbox[2] - person_bbox[0]) * 0.5
        y_aligned = abs(gear_bottom - person_top) < (person_bbox[3] - person_bbox[1]) * 0.3
        
        return x_aligned and y_aligned
    
    def annotate_frame(self, frame: np.ndarray, detections: sv.Detections,
                      compliance_status: Dict) -> np.ndarray:
        """
        Annotate frame with bounding boxes and labels.
        
        Args:
            frame: Input frame
            detections: Detections object
            compliance_status: Compliance check results
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Get labels for each detection
        labels = []
        colors = []
        
        for i, class_id in enumerate(detections.class_id):
            class_name = CLASS_NAMES.get(class_id, 'Unknown')
            confidence = detections.confidence[i]
            label = f"{class_name} {confidence:.2f}"
            labels.append(label)
            
            # Determine color based on class and compliance
            if class_name in ['No-Helmet', 'No-Vest', 'No-Mask', 'No-Hard-Hat']:
                colors.append(COLORS['violation'])
            elif class_name in ['Helmet', 'Safety-Vest', 'Mask', 'Hard-Hat']:
                colors.append(COLORS['safe'])
            else:
                colors.append(CLASS_COLORS.get(class_name, COLORS['default']))
        
        # Create custom color lookup
        if len(colors) > 0:
            # Use supervision annotator with custom colors
            annotated_frame = box_annotator.annotate(annotated_frame, detections)
            annotated_frame = label_annotator.annotate(annotated_frame, detections, labels=labels)
        
        # Add compliance status overlay
        if not compliance_status['is_compliant']:
            # Add red warning banner
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (0, 0), (annotated_frame.shape[1], 60), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
            cv2.putText(annotated_frame, "⚠️ VIOLATION DETECTED ⚠️", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            # Add green safe banner
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (0, 0), (annotated_frame.shape[1], 60), (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.5, annotated_frame, 0.5, 0, annotated_frame)
            cv2.putText(annotated_frame, "✅ SAFETY COMPLIANT", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        return annotated_frame
    
    def log_violation(self, frame_number: int, timestamp: datetime,
                     compliance_status: Dict, video_source: str = "Unknown") -> None:
        """Log violation to history."""
        if not compliance_status['is_compliant']:
            for violation in compliance_status['violations']:
                self.violation_history.append({
                    'frame_number': frame_number,
                    'timestamp': timestamp,
                    'violation_type': violation['type'],
                    'severity': violation['severity'],
                    'confidence': violation['confidence'],
                    'persons_detected': self.persons_detected,
                    'video_source': video_source
                })
            self.violation_count += len(compliance_status['violations'])
        else:
            self.safe_count += 1
    
    def get_violations_dataframe(self) -> pd.DataFrame:
        """Get violations as pandas DataFrame."""
        if not self.violation_history:
            return pd.DataFrame(columns=['frame_number', 'timestamp', 'violation_type', 
                                        'severity', 'confidence', 'persons_detected', 'video_source'])
        return pd.DataFrame(self.violation_history)
    
    def reset_stats(self) -> None:
        """Reset all statistics."""
        self.violation_history = []
        self.frame_count = 0
        self.safe_count = 0
        self.violation_count = 0
        self.persons_detected = 0
    
    def get_metrics(self) -> Dict:
        """Get current detection metrics."""
        total = self.safe_count + self.violation_count
        compliance_rate = (self.safe_count / total * 100) if total > 0 else 100
        
        return {
            'total_frames': self.frame_count,
            'persons_detected': self.persons_detected,
            'safe_count': self.safe_count,
            'violation_count': self.violation_count,
            'compliance_rate': compliance_rate,
            'total_detections': total
        }


def create_compliance_pie_chart(safe_count: int, violation_count: int) -> go.Figure:
    """Create a pie chart for compliance statistics."""
    labels = ['Safe', 'Violations']
    values = [safe_count, violation_count]
    colors = ['#00ff00', '#ff0000']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=colors,
        textinfo='label+percent',
        textposition='outside'
    )])
    
    fig.update_layout(
        title_text="Safety Compliance Distribution",
        showlegend=True,
        height=350,
        template='plotly_dark'
    )
    
    return fig


def create_violations_bar_chart(violations_df: pd.DataFrame) -> go.Figure:
    """Create a bar chart for violation types."""
    if violations_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No violations detected",
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    violation_counts = violations_df['violation_type'].value_counts()
    
    fig = go.Figure(data=[go.Bar(
        x=violation_counts.index,
        y=violation_counts.values,
        marker_color='red',
        text=violation_counts.values,
        textposition='auto'
    )])
    
    fig.update_layout(
        title_text="Violations by Type",
        xaxis_title="Violation Type",
        yaxis_title="Count",
        height=350,
        template='plotly_dark'
    )
    
    return fig


def create_severity_chart(violations_df: pd.DataFrame) -> go.Figure:
    """Create a chart showing violation severity distribution."""
    if violations_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No violations detected",
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    severity_counts = violations_df['severity'].value_counts()
    severity_colors = {'HIGH': 'red', 'MEDIUM': 'orange', 'LOW': 'yellow'}
    colors = [severity_colors.get(s, 'gray') for s in severity_counts.index]
    
    fig = go.Figure(data=[go.Bar(
        x=severity_counts.index,
        y=severity_counts.values,
        marker_color=colors,
        text=severity_counts.values,
        textposition='auto'
    )])
    
    fig.update_layout(
        title_text="Violations by Severity",
        xaxis_title="Severity Level",
        yaxis_title="Count",
        height=350,
        template='plotly_dark'
    )
    
    return fig


def resize_frame(frame: np.ndarray, max_width: int = 640, max_height: int = 480) -> np.ndarray:
    """Resize frame for faster processing while maintaining aspect ratio."""
    height, width = frame.shape[:2]
    
    if width > max_width or height > max_height:
        scale = min(max_width / width, max_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(frame, (new_width, new_height))
    
    return frame


def get_video_writer(output_path: str, fps: float, frame_size: Tuple[int, int]) -> cv2.VideoWriter:
    """Initialize video writer with appropriate codec."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, frame_size)


def process_video_file(video_path: str, detector: SafetyDetector, 
                      rule_type: str = 'construction_worker',
                      progress_callback=None) -> Tuple[List[np.ndarray], pd.DataFrame]:
    """
    Process entire video file.
    
    Args:
        video_path: Path to input video
        detector: SafetyDetector instance
        rule_type: Type of safety rule
        progress_callback: Optional callback for progress updates
        
    Returns:
        Tuple of (annotated_frames, violations_dataframe)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    annotated_frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Skip frames for performance (optional)
        if frame_count % (PERFORMANCE['frame_skip'] + 1) != 0:
            continue
        
        # Resize for faster processing
        processed_frame = resize_frame(frame, 
                                     PERFORMANCE['resize_width'], 
                                     PERFORMANCE['resize_height'])
        
        # Detect and check compliance
        detections = detector.detect(processed_frame)
        compliance = detector.check_compliance(detections, rule_type)
        
        # Log violations
        detector.log_violation(
            frame_count,
            datetime.now(),
            compliance,
            video_path
        )
        
        # Annotate frame
        annotated = detector.annotate_frame(processed_frame, detections, compliance)
        annotated_frames.append(annotated)
        
        # Update progress
        if progress_callback and frame_count % 10 == 0:
            progress = frame_count / total_frames
            progress_callback(progress, frame_count, total_frames)
    
    cap.release()
    
    violations_df = detector.get_violations_dataframe()
    return annotated_frames, violations_df


def save_annotated_video(frames: List[np.ndarray], output_path: str, fps: float = 20.0) -> None:
    """Save annotated frames as video file."""
    if not frames:
        return
    
    height, width = frames[0].shape[:2]
    writer = get_video_writer(output_path, fps, (width, height))
    
    for frame in frames:
        writer.write(frame)
    
    writer.release()


def download_csv(df: pd.DataFrame, filename: str = "violations_report.csv") -> str:
    """Convert DataFrame to CSV download link for Streamlit."""
    csv = df.to_csv(index=False)
    return csv


def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get information about a YOLO model."""
    from config import MODEL_CONFIG
    return MODEL_CONFIG.get(model_name, MODEL_CONFIG['yolov8n'])

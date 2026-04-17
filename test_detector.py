"""
SafeGear AI - Standalone Detector Test Script
Test the detection pipeline without Streamlit.

Usage:
    python test_detector.py --image path/to/image.jpg
    python test_detector.py --video path/to/video.mp4
    python test_detector.py --webcam
"""

import argparse
import cv2
import time
from pathlib import Path
from datetime import datetime

from utils import SafetyDetector, save_annotated_video
from config import SAFETY_RULES


def test_image(image_path: str, model: str = 'yolov8n.pt', conf: float = 0.45):
    """Test detection on a single image."""
    print(f"\n🖼️  Testing image: {image_path}")
    
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    # Initialize detector
    print(f"🤖 Loading model: {model}")
    detector = SafetyDetector(model_path=model, conf_threshold=conf)
    
    # Run detection
    print("🔍 Running detection...")
    start_time = time.time()
    
    detections = detector.detect(frame)
    compliance = detector.check_compliance(detections, 'construction_worker')
    
    elapsed = time.time() - start_time
    
    # Annotate
    annotated = detector.annotate_frame(frame, detections, compliance)
    
    # Save output
    output_path = f"test_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(output_path, annotated)
    
    # Print results
    print(f"\n📊 Results:")
    print(f"   Processing time: {elapsed:.3f}s")
    print(f"   Detections: {len(detections)}")
    print(f"   Persons: {compliance.get('persons_detected', 0)}")
    print(f"   Violations: {len(compliance.get('violations', []))}")
    print(f"   Compliant: {compliance['is_compliant']}")
    
    if compliance['violations']:
        print(f"\n⚠️  Violations detected:")
        for v in compliance['violations']:
            print(f"   - {v['type']} (Severity: {v['severity']}, Conf: {v['confidence']:.2f})")
    
    print(f"\n✅ Output saved: {output_path}")
    
    # Display
    cv2.imshow("SafeGear AI - Test Output", annotated)
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_video(video_path: str, model: str = 'yolov8n.pt', conf: float = 0.45, 
               max_frames: int = None):
    """Test detection on video file."""
    print(f"\n🎬 Testing video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return
    
    # Initialize detector
    print(f"🤖 Loading model: {model}")
    detector = SafetyDetector(model_path=model, conf_threshold=conf)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n📹 Video Info:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Total frames: {total_frames}")
    
    # Process frames
    frame_count = 0
    start_time = time.time()
    annotated_frames = []
    
    print("\n⏳ Processing... (Press 'q' to stop early)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        if max_frames and frame_count > max_frames:
            break
        
        # Process every 2nd frame for speed
        if frame_count % 2 == 0:
            # Resize for faster processing
            frame = cv2.resize(frame, (640, 480))
            
            # Detect
            detections = detector.detect(frame)
            compliance = detector.check_compliance(detections, 'construction_worker')
            
            # Annotate
            annotated = detector.annotate_frame(frame, detections, compliance)
            
            # Add frame counter
            cv2.putText(annotated, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            annotated_frames.append(annotated)
            
            # Show progress
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"   Progress: {frame_count}/{total_frames} frames ({progress:.1f}%)")
        
        # Early exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n⚠️  Stopped by user")
            break
    
    cap.release()
    
    # Save output
    if annotated_frames:
        output_path = f"test_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        print(f"\n💾 Saving annotated video...")
        save_annotated_video(annotated_frames, output_path)
        
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        
        # Print summary
        print(f"\n📊 Summary:")
        print(f"   Frames processed: {frame_count}")
        print(f"   Processing time: {elapsed:.2f}s")
        print(f"   Average FPS: {avg_fps:.1f}")
        print(f"   Total violations: {detector.violation_count}")
        print(f"   Output saved: {output_path}")


def test_webcam(model: str = 'yolov8n.pt', conf: float = 0.45):
    """Test real-time detection with webcam."""
    print("\n📷 Starting webcam test...")
    print("   Press 'q' to exit\n")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not access webcam")
        return
    
    # Set properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Initialize detector
    print(f"🤖 Loading model: {model}")
    detector = SafetyDetector(model_path=model, conf_threshold=conf)
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame")
            break
        
        frame_count += 1
        
        # Process every 2nd frame
        if frame_count % 2 == 0:
            # Detect
            detections = detector.detect(frame)
            compliance = detector.check_compliance(detections, 'construction_worker')
            
            # Annotate
            annotated = detector.annotate_frame(frame, detections, compliance)
            
            # Calculate FPS
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Add FPS overlay
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show
            cv2.imshow("SafeGear AI - Webcam Test", annotated)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print summary
    elapsed = time.time() - start_time
    avg_fps = frame_count / elapsed if elapsed > 0 else 0
    
    print(f"\n📊 Webcam Test Summary:")
    print(f"   Total frames: {frame_count}")
    print(f"   Duration: {elapsed:.2f}s")
    print(f"   Average FPS: {avg_fps:.1f}")
    print(f"   Violations detected: {detector.violation_count}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Test SafeGear AI detection pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_detector.py --image worker.jpg
  python test_detector.py --video construction_site.mp4 --max-frames 100
  python test_detector.py --webcam --model yolov8s.pt
        """
    )
    
    parser.add_argument('--image', type=str, help='Path to test image')
    parser.add_argument('--video', type=str, help='Path to test video')
    parser.add_argument('--webcam', action='store_true', help='Use webcam')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='YOLO model to use (default: yolov8n.pt)')
    parser.add_argument('--conf', type=float, default=0.45,
                       help='Confidence threshold (default: 0.45)')
    parser.add_argument('--max-frames', type=int,
                       help='Maximum frames to process (video only)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🦺 SafeGear AI - Detection Test")
    print("=" * 60)
    
    if args.image:
        test_image(args.image, args.model, args.conf)
    elif args.video:
        test_video(args.video, args.model, args.conf, args.max_frames)
    elif args.webcam:
        test_webcam(args.model, args.conf)
    else:
        parser.print_help()
        print("\n❌ Please specify --image, --video, or --webcam")


if __name__ == "__main__":
    main()

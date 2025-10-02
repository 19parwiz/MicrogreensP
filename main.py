# AgroTech Recognition System
# Professional microgreens and face recognition system
# Supports multiple detection modes and camera sources

import cv2
import argparse
import os
import time
from dotenv import load_dotenv

from src.camera import Camera
from src.face_recognition_module import FaceRecognition
from src.plant_recognition_module import PlantRecognition
from src.enhanced_plant_recognition import EnhancedPlantRecognition

def _run_recognition(mode, recognizer, cam):
    """Main recognition loop for all detection modes"""
    if cam is None or cam.cap is None or not cam.cap.isOpened():
        print("ERROR: Camera is not opened. Please check camera connection.")
        return

    print(f"\nStarting {mode} recognition...")
    print("Controls: Press 'q' to quit")

    prev_time = time.time()
    frame_count = 0
    
    while True:
        frame = cam.read()
        if frame is None:
            print("WARNING: Failed to grab frame. Camera may be disconnected.")
            break

        result_frame = recognizer.predict(frame)

        # Calculate and display FPS
        curr_time = time.time()
        fps = 1 / max((curr_time - prev_time), 1e-6)
        prev_time = curr_time
        frame_count += 1
        
        cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(result_frame, f"Frame: {frame_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("AgroTech Recognition System", result_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print(f"\nSession ended. Processed {frame_count} frames.")
    cam.release()
    cv2.destroyAllWindows()


def run_face_recognition(cam):
    recognizer = FaceRecognition()
    _run_recognition("face", recognizer, cam)


def run_plant_recognition(cam, shrink_ratio=0.6, conf=0.25, display_conf=0.35, show_no_detection_text=True, min_area_ratio=0.005, min_side_px=20, persist_frames=2, persist_iou=0.5, safe_mode=False):
    recognizer = PlantRecognition(
        shrink_ratio=shrink_ratio,
        conf=conf,
        display_conf=display_conf,
        show_no_detection_text=show_no_detection_text,
        min_area_ratio=min_area_ratio,
        min_side_px=min_side_px,
        persist_frames=persist_frames,
        persist_iou=persist_iou,
        safe_mode=safe_mode,
    )
    _run_recognition("plant", recognizer, cam)


def run_enhanced_plant_recognition(cam, detection_mode="smart", conf=0.25, max_detections=6, min_detection_size=0.4):
    """Enhanced plant recognition with multiple detection modes"""
    print(f"\nInitializing Enhanced Plant Recognition...")
    print(f"Mode: {detection_mode}")
    print(f"Confidence: {conf}")
    print(f"Max detections: {max_detections}")
    
    recognizer = EnhancedPlantRecognition(
        conf=conf,
        detection_mode=detection_mode,
        max_detections=max_detections,
        min_detection_size=min_detection_size
    )
    _run_recognition(f"Enhanced Plant Detection ({detection_mode.title()})", recognizer, cam)


def choose_mode():
    """Interactive mode selection menu"""
    print("\nAgroTech Recognition System v2.0")
    print("\nAvailable Detection Modes:")
    print("\n1. Face Recognition")
    print("   - Detect and recognize faces")
    print("   - Uses trained face recognition model")
    
    print("\n2. Plant Recognition (Original)")
    print("   - Single microgreen detection")
    print("   - Original detection algorithm")
    
    print("\n3. Enhanced Multi-Microgreens (Smart) [RECOMMENDED]")
    print("   - Multiple microgreens detection")
    print("   - Advanced false positive filtering")
    print("   - Best for production deployment")
    
    print("\n4. Enhanced Multi-Microgreens (Precise)")
    print("   - Tight bounding boxes around microgreens")
    print("   - Multiple detection support")
    
    print("\n5. Enhanced Multi-Microgreens (Basic)")
    print("   - Simple multi-detection mode")
    print("   - Good for testing")
    
    print("\nq. Quit")
    
    while True:
        choice = input("Select mode (1-5 or q): ").strip().lower()
        if choice == "1":
            return "face"
        elif choice == "2":
            return "plant"
        elif choice == "3":
            return "enhanced_smart"
        elif choice == "4":
            return "enhanced_precise"
        elif choice == "5":
            return "enhanced_multi"
        elif choice == "q":
            return None
        else:
            print("Invalid choice. Please enter 1-5 or q.")


def choose_camera():
    """Interactive camera source selection"""
    env_url = os.getenv("CAMERA_URL", "").strip()
    
    print("\nCamera Source Selection")
    print("\nAvailable Camera Sources:")
    print("\n1. Local Camera (USB/Built-in) - Index 0")
    print("2. Local Camera (USB/Built-in) - Index 1")
    
    if env_url:
        print(f"3. IP Camera from .env file")
        print(f"   URL: {env_url}")
    else:
        print("3. IP Camera from .env file (not configured)")
    
    print("4. Custom RTSP/HTTP URL")
    print("q. Cancel")
    
    while True:
        choice = input("Select camera source (1-4 or q): ").strip().lower()
        
        if choice == "1":
            print("Selected: Local Camera (Index 0)")
            return 0
        elif choice == "2":
            print("Selected: Local Camera (Index 1)")
            return 1
        elif choice == "3":
            if env_url:
                print(f"Selected: IP Camera ({env_url})")
                return env_url
            else:
                print("ERROR: CAMERA_URL not set in .env file.")
                print("Defaulting to Local Camera (Index 0)")
                return 0
        elif choice == "4":
            url = input("Enter RTSP/HTTP URL: ").strip()
            if url:
                print(f"Selected: Custom URL ({url})")
                return url
            else:
                print("No URL entered. Defaulting to Local Camera (Index 0)")
                return 0
        elif choice == "q":
            print("Camera selection cancelled.")
            return 0
        else:
            print("Invalid choice. Please enter 1-4 or q.")


if __name__ == "__main__":
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="AgroTech Recognition System v2.0 - Professional microgreens and face recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Interactive mode
  python main.py --mode enhanced_smart --camera 0  # Smart detection with camera 0
  python main.py --mode face --camera 1            # Face recognition with camera 1
  python main.py --mode enhanced_smart --conf 0.3  # Higher confidence threshold

Detection Modes:
  face            - Face recognition
  plant           - Original single microgreen detection
  enhanced_smart  - Multi-microgreens with smart filtering (RECOMMENDED)
  enhanced_precise- Multi-microgreens with precise bounding boxes
  enhanced_multi  - Basic multi-microgreens detection
        """
    )
    
    # Main options
    parser.add_argument("--mode", type=str, 
                       help="Detection mode (face, plant, enhanced_smart, enhanced_precise, enhanced_multi)")
    parser.add_argument("--camera", type=str, 
                       help="Camera source: index (0,1), 'env' for .env file, or RTSP URL")
    
    # Detection parameters
    parser.add_argument("--conf", type=float, default=0.25, 
                       help="Model confidence threshold (0.1-1.0, default: 0.25)")
    parser.add_argument("--max-detections", dest="max_detections", type=int, default=6,
                       help="Maximum microgreens to detect (1-20, default: 6)")
    parser.add_argument("--min-detection-size", dest="min_detection_size", type=float, default=0.4,
                       help="Minimum detection size for subdivision (0.1-1.0, default: 0.4)")
    
    # Original mode parameters (for backward compatibility)
    parser.add_argument("--shrink", type=float, default=0.6, 
                       help="Box size ratio for original mode (0.05-1.0, default: 0.6)")
    parser.add_argument("--display-conf", dest="display_conf", type=float, default=0.35,
                       help="Min confidence to display boxes (original mode, default: 0.35)")
    parser.add_argument("--no-text", action="store_true", 
                       help="Hide 'No microgreens' text when none detected")
    parser.add_argument("--min-area", dest="min_area", type=float, default=0.005,
                       help="Minimum box area ratio (original mode, default: 0.005)")
    parser.add_argument("--min-side", dest="min_side", type=int, default=20,
                       help="Minimum side length in pixels (original mode, default: 20)")
    parser.add_argument("--persist-frames", dest="persist_frames", type=int, default=2,
                       help="Frames required before display (original mode, default: 2)")
    parser.add_argument("--persist-iou", dest="persist_iou", type=float, default=0.5,
                       help="IoU threshold for persistence (original mode, default: 0.5)")
    parser.add_argument("--safe-mode", dest="safe_mode", action="store_true",
                       help="Bypass filters in original mode")
    
    args = parser.parse_args()

    print("\nAgroTech Recognition System v2.0")
    print("Professional Microgreens Detection")
    
    # Determine camera source
    if args.camera:
        camera_arg = args.camera.strip()
        if camera_arg.lower() == "env":
            camera_source = os.getenv("CAMERA_URL", "0")
            print(f"\nUsing camera from .env: {camera_source}")
        else:
            camera_source = int(camera_arg) if camera_arg.isdigit() else camera_arg
            print(f"\nUsing specified camera: {camera_source}")
    else:
        camera_source = choose_camera()

    # Initialize camera with error handling
    print(f"\nInitializing camera...")
    cam = Camera(index=camera_source)

    if not cam.open():
        print("ERROR: Failed to open camera.")
        print("Troubleshooting:")
        print("- Check camera connection")
        print("- Ensure camera is not used by another application")
        print("- Try a different camera index")
        
        retry_attempts = 3
        for attempt in range(retry_attempts):
            print(f"\nRetry attempt {attempt + 1}/{retry_attempts}")
            new_source = choose_camera()
            cam = Camera(index=new_source)
            
            if cam.open():
                print("SUCCESS: Camera opened successfully!")
                break
                
            if attempt < retry_attempts - 1:
                continue_retry = input("Continue trying? (y/n): ").strip().lower()
                if continue_retry != "y":
                    break
        else:
            print("\nERROR: All camera initialization attempts failed.")
            print("Please check your camera setup and try again.")
            exit(1)
    else:
        print("SUCCESS: Camera initialized successfully!")

    # Determine mode: CLI > menu
    mode = args.mode if args.mode else choose_mode()

    if mode == "face":
        run_face_recognition(cam)
    elif mode == "plant":
        run_plant_recognition(
            cam,
            shrink_ratio=max(0.05, min(1.0, float(args.shrink))),
            conf=max(0.0, min(1.0, float(args.conf))),
            display_conf=max(0.0, min(1.0, float(args.display_conf))),
            show_no_detection_text=not args.no_text,
            min_area_ratio=max(0.0, min(1.0, float(args.min_area))),
            min_side_px=max(0, int(args.min_side)),
            persist_frames=max(1, int(args.persist_frames)),
            persist_iou=max(0.0, min(1.0, float(args.persist_iou))),
            safe_mode=bool(args.safe_mode),
        )
    elif mode == "enhanced_smart":
        run_enhanced_plant_recognition(
            cam,
            detection_mode="smart",
            conf=max(0.05, min(1.0, float(args.conf))),
            max_detections=max(1, min(20, int(args.max_detections))),
            min_detection_size=max(0.1, min(1.0, float(args.min_detection_size)))
        )
    elif mode == "enhanced_precise":
        run_enhanced_plant_recognition(
            cam,
            detection_mode="precise",
            conf=max(0.05, min(1.0, float(args.conf))),
            max_detections=max(1, min(20, int(args.max_detections))),
            min_detection_size=max(0.1, min(1.0, float(args.min_detection_size)))
        )
    elif mode == "enhanced_multi":
        run_enhanced_plant_recognition(
            cam,
            detection_mode="multi",
            conf=max(0.05, min(1.0, float(args.conf))),
            max_detections=max(1, min(20, int(args.max_detections))),
            min_detection_size=max(0.1, min(1.0, float(args.min_detection_size)))
        )
    else:
        print("\nExiting AgroTech Recognition System.")
        print("Thank you for using our professional detection system!")
        
    print("\nSession completed successfully.")

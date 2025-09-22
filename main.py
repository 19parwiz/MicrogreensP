import cv2
import argparse
import os
import time

from src.camera import Camera
from src.face_recognition_module import FaceRecognition
from src.plant_recognition_module import PlantRecognition
from dotenv import load_dotenv
from ultralytics import YOLO


# Shared recognition loop (used for both face and plant recognition)
def _run_recognition(mode, recognizer, cam):
    if not cam.open():
        print("Could not open camera")
        return

    print(f"Running in {mode} recognition mode... Press 'q' to quit.")

    # Optional: track FPS
    prev_time = time.time()
    while True:
        frame = cam.read()
        if frame is None:
            print("Failed to grab frame")
            break

        result_frame = recognizer.predict(frame)

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("AgroTech Recognition", result_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


# Wrapper for face recognition
def run_face_recognition(cam):
    recognizer = FaceRecognition()
    _run_recognition("face", recognizer, cam)


# Wrapper for plant recognition
def run_plant_recognition(cam):
    recognizer = PlantRecognition()
    _run_recognition("plant", recognizer, cam)


# Menu to choose mode if --mode argument is not provided
def choose_mode():
    print("\n=== AgroTech Recognition System ===")
    print("1. Face Recognition")
    print("2. Plant Recognition")
    print("q. Quit")

    choice = input("Enter choice: ").strip()
    if choice == "1":
        return "face"
    elif choice == "2":
        return "plant"
    else:
        return None


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="AgroTech Recognition System")
    parser.add_argument("--mode", type=str, help="Mode: face or plant")
    parser.add_argument("--camera", type=str, help="Camera index (0,1,2) or RTSP URL")
    args = parser.parse_args()

    # Determine camera source: command line > .env > default 0
    camera_arg = args.camera.strip() if args.camera else os.getenv("CAMERA_URL", "0")
    camera_source = int(camera_arg) if camera_arg.isdigit() else camera_arg
    cam = Camera(index=camera_source)

    # Determine mode: command line > menu
    mode = args.mode if args.mode else choose_mode()

    if mode == "face":
        run_face_recognition(cam)
    elif mode == "plant":
        run_plant_recognition(cam)
    else:
        print("Exiting program.")

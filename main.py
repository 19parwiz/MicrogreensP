import cv2
import argparse
from modules.camera import Camera
from modules.face_recognition_module import FaceRecognition
from modules.plant_recognition_module import PlantRecognition

def main(mode):
    cam = Camera()
    if not cam.open():
        print("Could not open camera")
        return

    if mode == "face":
        recognizer = FaceRecognition()
    elif mode == "plant":
        recognizer = PlantRecognition()
    else:
        print("Invalid mode! Use 'face' or 'plant'.")
        return

    print(f"Running in {mode} recognition mode... Press 'q' to quit.")

    while True:
        # Read frame from camera - FIXED THIS LINE
        frame = cam.read()
        if frame is None:
            print("Failed to grab frame")
            break

        # Process the frame in here
        result_frame = recognizer.predict(frame)
        
        # Display the result
        cv2.imshow("AgroTech Recognition", result_frame)
        
        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgroTech Recognition System")
    parser.add_argument("--mode", type=str, required=True, help="Mode: face or plant")
    args = parser.parse_args()
    main(args.mode)
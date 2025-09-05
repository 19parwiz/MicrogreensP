import cv2
import argparse
import os

from modules.camera import Camera
from modules.face_recognition_module import FaceRecognition
from modules.plant_recognition_module import PlantRecognition
from dotenv import load_dotenv



# Main function: handles camera input and runs recognition based on the mode
def main(mode, cam):
    #cam = Camera()
    if not cam.open():
        print("Could not open camera")
        return

    # choose based on mode
    if mode == "face":
        recognizer = FaceRecognition()
    elif mode == "plant":
        recognizer = PlantRecognition()
    else:
        print("Invalid mode! Use 'face' or 'plant'.")
        return

    print(f"Running in {mode} recognition mode... Press 'q' to quit.")

    while True:
        # Read frame from camera 
        frame = cam.read()
        if frame is None:
            print("Failed to grab frame")
            break

        # Process the frame in here
        result_frame = recognizer.predict(frame)
        
       
        cv2.imshow("AgroTech Recognition", result_frame)
        
        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="AgroTech Recognition System")
    parser.add_argument("--mode", type=str, required=True, help="Mode: face or plant")
    parser.add_argument("--camera", type=str, help="Camera index (0,1,2) or RTSP URL")

    args = parser.parse_args()

    # Priority: command line > .env > default 0
    camera_arg = args.camera.strip() if args.camera else os.getenv("CAMERA_URL", "0")
    camera_source = int(camera_arg) if camera_arg.isdigit() else camera_arg

    cam = Camera(index=camera_source)
    main(args.mode, cam)


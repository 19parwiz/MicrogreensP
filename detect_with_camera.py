import cv2
from src.camera import Camera
from src.plant_recognition_module import PlantRecognition

# Initialize recognition module with smaller boxes (tune shrink_ratio as needed)
recognizer = PlantRecognition(
    weights_path="D:/AgroTech/runs/detect/train9/weights/best.pt",
    conf=0.25,
    iou=0.45,
    device=None,
    shrink_ratio=0.6,
)

# Initialize camera
camera = Camera(index=0)

if not camera.open():
    print("Error: Could not open camera")
    exit()

print("Camera opened successfully. Press 'q' to quit.")

while True:
    frame = camera.read()
    if frame is None:
        print("No frame captured")
        break

    # Run detection and draw our smaller boxes
    annotated_frame = recognizer.predict(frame)

    # Resize for better visibility
    resized_frame = cv2.resize(annotated_frame, (800, 600))

    # Show the video stream
    cv2.imshow("Microgreens Detection", resized_frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
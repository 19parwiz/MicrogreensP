import cv2
from ultralytics import YOLO
from src.camera import Camera

# Load trained YOLO model
model = YOLO("D:/AgroTech/runs/detect/train9/weights/best.pt")

# Initialize camera (default index 0)
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

    # Run YOLO detection
    results = model.predict(frame, conf=0.25)

    # Annotate frame
    annotated_frame = results[0].plot()

    # Resize for better visibility (optional)
    resized_frame = cv2.resize(annotated_frame, (800, 600))

    # Show the video stream
    cv2.imshow("Microgreens Detection", resized_frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

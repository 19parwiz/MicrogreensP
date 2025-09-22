from ultralytics import YOLO
import cv2
import glob

# Load trained model
model = YOLO("D:/AgroTech/runs/detect/train9/weights/best.pt")

# Path to test images
test_path = "D:/AgroTech/test_images/*.*"

for img_path in glob.glob(test_path):
    results = model.predict(source=img_path, conf=0.25, save=False, imgsz=640)

    res = results[0]
    img = res.plot()

    # Resize to fixed window (e.g., 800x600)
    resized = cv2.resize(img, (800, 600))

    cv2.imshow("Prediction", resized)

    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):  # quit if 'q' is pressed
        break

cv2.destroyAllWindows()

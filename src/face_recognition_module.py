import cv2
import pickle
import os

class FaceRecognition:
    def __init__(self):
        # Paths
        self.model_path = os.path.join("models", "face_trainer.yml")
        self.labels_path = os.path.join("models", "labels.pkl")

        # Load cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        # Load recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read(self.model_path)

        # Load labels
        with open(self.labels_path, "rb") as f:
            self.labels = pickle.load(f)
        self.labels = {v: k for k, v in self.labels.items()}  # reverse dict: id → name


    def predict(self, frame):    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            id_, conf = self.recognizer.predict(roi_gray)

            if conf < 70:  # smaller = more confident
                name = self.labels.get(id_, "Unknown")
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        return frame


import cv2

class Camera:
    def __init__(self, index=0):
        self.index = index
        self.cap = None

    """def open(self):
    # Try MSMF first
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_MSMF)
        if not self.cap.isOpened():
            print("MSMF backend failed, trying DirectShow...")
        # Try DirectShow as fallback
            self.cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        return self.cap.isOpened()

    """
    # open the camera (works for local or IP)
    def open(self):
        # If it's an IP Camera( RTSP/HTTP URL), just try to connect
        if isinstance(self.index, str) and (self.index.startswith("rtsp://") or self.index.startswith("http://")):
            print(f"Opening IP camera: {self.index}")
            self.cap = cv2.VideoCapture(self.index)
            return self.cap.isOpened()

        # local webcam
        print(f"Opening local camera index: {self.index}")
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_MSMF)
        if not self.cap.isOpened():
            print("MSMF backend failed, trying DirectShow...")
            self.cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)

        return self.cap.isOpened()

    # grab a frame from camera
    def read(self):
        if self.cap is None or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    
    def release(self):
        if self.cap is not None:
            self.cap.release()

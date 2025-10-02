# src/camera.py
# [MODIFIED] Improved camera open logic to:
#  - accept int or digit-strings for local cameras
#  - accept RTSP/HTTP strings for IP cameras
#  - try multiple backends (DSHOW/MSMF for local; FFMPEG then default for IP)
#  - do a short retry loop for IP cameras to wait for first frame
#  - be resilient to failures and release intermediate captures

import cv2
import time

class Camera:
    def __init__(self, index=0, open_timeout=5.0):
        """
        index: int (0,1,...) or RTSP/HTTP URL string
        open_timeout: seconds to wait for first frame on IP streams
        """
        self.index = index
        self.cap = None
        self.open_timeout = float(open_timeout)

    def _try_open(self, source, backend=None):
        """Try to create a VideoCapture with optional backend. Return capture or None."""
        try:
            if backend is None:
                cap = cv2.VideoCapture(source)
            else:
                cap = cv2.VideoCapture(source, backend)
            return cap
        except Exception as e:
            # keep simple -- print small diagnostic
            print(f"[Camera] _try_open exception for backend {backend}: {e}")
            return None

    def open(self):
        """
        Open camera. Returns True if opened and a first frame was read (for IP).
        """
        # If camera index is a digit string, convert to int (local camera)
        if isinstance(self.index, str) and self.index.strip().isdigit():
            self.index = int(self.index.strip())

        # IP camera (rtsp/http)
        if isinstance(self.index, str) and (self.index.startswith("rtsp://") or self.index.startswith("http://")):
            url = self.index
            print(f"Opening IP camera: {url}")

            # Try FFMPEG backend first (if available), then default
            ffmpeg_backend = getattr(cv2, "CAP_FFMPEG", None)
            backends = [ffmpeg_backend, None] if ffmpeg_backend is not None else [None]

            for backend in backends:
                cap = self._try_open(url, backend)
                if cap is None:
                    continue

                deadline = time.time() + self.open_timeout
                while time.time() < deadline:
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            self.cap = cap
                            print("[Camera] IP stream opened and first frame received.")
                            return True
                    # small sleep to avoid busy loop
                    time.sleep(0.3)

                # didn't get a frame in time -> release and try next backend
                try:
                    cap.release()
                except Exception:
                    pass

            print("[Camera] Failed to open IP stream or receive frame within timeout.")
            return False

        # Local webcam (index is int here)
        print(f"Opening local camera index: {self.index}")
        # Try DirectShow first (works well on many Windows setups), then MSMF, then default
        backends_local = []
        dshow = getattr(cv2, "CAP_DSHOW", None)
        msmf = getattr(cv2, "CAP_MSMF", None)
        # prefer DSHOW then MSMF then default None
        if dshow is not None:
            backends_local.append(dshow)
        if msmf is not None:
            backends_local.append(msmf)
        backends_local.append(None)

        for backend in backends_local:
            cap = self._try_open(self.index, backend)
            if cap is None:
                continue

            # if open returns true immediately, accept it
            if cap.isOpened():
                self.cap = cap
                print(f"[Camera] Opened with backend {backend}.")
                return True

            # some backends might require a read attempt
            try:
                ret, frame = cap.read()
                if ret and frame is not None:
                    self.cap = cap
                    print(f"[Camera] Opened after read with backend {backend}.")
                    return True
            except Exception:
                pass

            try:
                cap.release()
            except Exception:
                pass

        print("[Camera] Could not open local camera with any backend.")
        return False

    def read(self):
        """Return one BGR frame or None."""
        if self.cap is None or not self.cap.isOpened():
            return None
        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                return None
            return frame
        except Exception:
            return None

    def release(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

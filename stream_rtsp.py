import subprocess
import shutil
import sys

class RTSPPusher:
    def __init__(self, rtsp_url: str, width: int, height: int, fps: int = 25, bitrate: str = "2500k"):
        self.rtsp_url = rtsp_url
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.bitrate = str(bitrate)
        self.proc = None

    def start(self):
        if not shutil.which("ffmpeg"):
            print("[RTSP] ERROR: ffmpeg not found in PATH. Install FFmpeg.", file=sys.stderr)
            return False

        cmd = [
            "ffmpeg",
            "-re",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "-",
            "-an",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-tune", "zerolatency",
            "-pix_fmt", "yuv420p",
            "-b:v", self.bitrate,
            "-g", str(max(self.fps, 1) * 2),
            "-rtsp_transport", "tcp",
            "-f", "rtsp",
            self.rtsp_url
        ]
        try:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            print(f"[RTSP] started -> {self.rtsp_url} ({self.width}x{self.height}@{self.fps}fps, {self.bitrate})")
            return True
        except Exception as e:
            print(f"[RTSP] ERROR: failed to start ffmpeg: {e}", file=sys.stderr)
            self.proc = None
            return False

    def push(self, frame_bgr):
        if not self.proc or not self.proc.stdin:
            return
        try:
            self.proc.stdin.write(frame_bgr.tobytes())
        except BrokenPipeError:
            print("[RTSP] WARN: ffmpeg pipe closed (BrokenPipe). Is server reachable?")
            self.stop()

    def stop(self):
        if self.proc:
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.terminate()
                self.proc.wait(timeout=3)
            except Exception:
                pass
            finally:
                self.proc = None
            print("[RTSP] stopped")

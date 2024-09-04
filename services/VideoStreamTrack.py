import asyncio
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
import cv2

class VideoStreamTrack(MediaStreamTrack):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        self.kind = "video"

    async def recv(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to read frame")
            await asyncio.sleep(0.1)  # Prevent tight loop in case of failure
            return None

        return frame

    async def stop(self):
        if self.cap.isOpened():
            self.cap.release()  # Release the camera when stopping
            print("Camera released")
        # Remove 'await' since the parent's stop method may not be async
        super().stop()  # Call the parent class's stop method (without 'await')
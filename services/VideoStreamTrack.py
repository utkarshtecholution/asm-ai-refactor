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
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
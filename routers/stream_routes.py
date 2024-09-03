import cv2
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from aiortc import RTCPeerConnection
from services.VideoStreamTrack import VideoStreamTrack
from settings.swagger_doc import (
    STREAM_TAGS
)
from Config import STREAMING_FRAME_SIZE
from ai_pipeline.main import frame_queue, new_item_event

router = APIRouter(
    tags=STREAM_TAGS
)

@router.get(
    path="/video_feed"
)
async def video_feed():
    pc = RTCPeerConnection()
    video_track = VideoStreamTrack()
    pc.addTrack(video_track)

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    async def stream_video():
        while True:
            frame = await video_track.recv()
            if frame is None:
                break
            
            frame_queue.put(frame)
            new_item_event.set() # Signal that a new item is available
            stream_frame = cv2.resize(frame, STREAMING_FRAME_SIZE, cv2.INTER_CUBIC)
            ret, buffer = cv2.imencode(".jpg", stream_frame)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )

    return StreamingResponse(
        stream_video(),
        media_type="multipart/x-mixed-replace;boundary=frame"
    )

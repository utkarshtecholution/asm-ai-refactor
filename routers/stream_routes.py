import cv2
import asyncio
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from aiortc import RTCPeerConnection
from services.VideoStreamTrack import VideoStreamTrack
from settings.swagger_doc import STREAM_TAGS
from Config import STREAMING_FRAME_SIZE
from ai_pipeline.main import frame_queue, new_item_event

router = APIRouter(tags=STREAM_TAGS)

@router.get("/video_feed")
async def video_feed():
    pc = RTCPeerConnection()
    video_track = VideoStreamTrack()
    pc.addTrack(video_track)

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    async def stream_video():
        try:
            while True:
                frame = await video_track.recv()
                if frame is None:
                    break
                frame_queue.append(frame)
                new_item_event.set() # Signal that a new item is available
                stream_frame = cv2.resize(frame, STREAMING_FRAME_SIZE, cv2.INTER_CUBIC)

                ret, buffer = cv2.imencode(".jpg", stream_frame)
                if not ret:
                    continue
                try:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                    )
                except:
                    print(f"Connection closed")
                    break  # Exit the loop if there's an issue with the connection

                await asyncio.sleep(0.01)  # Yield control to the event loop

        except asyncio.CancelledError:
            print("Stream was cancelled")

        finally:
            await video_track.stop()  # Ensure camera is released

    return StreamingResponse(
        stream_video(),
        media_type="multipart/x-mixed-replace;boundary=frame"
    )

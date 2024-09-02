import threading
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from services.camera_stream_configurator import CameraStreamConfigurator
from settings.swagger_doc import (
    OCR_TAGS
)

router = APIRouter(
    tags=OCR_TAGS
)

@router.get(
    path="/ocr_inference"
)
async def ocr_inference():
    camera_stream_object = CameraStreamConfigurator(0)
    camera_stream_object.stream_flag = False
    frame = camera_stream_object.org_frame
    if frame is None:
        return JSONResponse(content = {"status": "Failed"},  status_code =400)
    # threading.Thread(target = process_image_ocr, args = (frame, )).start()
    
    return JSONResponse(content={"status": "request submitted"}, status_code=200)
    
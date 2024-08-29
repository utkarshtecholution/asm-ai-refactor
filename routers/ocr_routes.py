import threading
from fastapi import APIRouter
from fastapi.responses import JSONResponse
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
    # camera_object.stream_flag = False
    # frame = camera_object.org_frame
    # if frame is None:
    #     return JSONResponse(content = {"status": "Failed"},  status_code =400)
    # threading.Thread(target = process_image_ocr, args = (frame, )).start()
    
    # return JSONResponse(content={"status": "request submitted"}, status_code=200)
    return True
    
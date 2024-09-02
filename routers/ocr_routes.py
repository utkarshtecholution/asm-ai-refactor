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
    return JSONResponse(content={"status": "request submitted"}, status_code=200)
    
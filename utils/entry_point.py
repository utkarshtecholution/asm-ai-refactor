import sys
import uvicorn
from settings.config import (
    UVICORN_APP,
    UVICORN_HOST,
    UVICORN_PORT,
    UVICORN_RELOAD,
    UVICORN_LOG_LEVEL
)

class EntryPoint:
    def __init__(self):
        pass

    def run(self):
        try:
            uvicorn.run(
                UVICORN_APP, 
                host=UVICORN_HOST, 
                port=UVICORN_PORT, 
                log_level=UVICORN_LOG_LEVEL,
                reload=UVICORN_RELOAD
                )
        finally:
            sys.exit()
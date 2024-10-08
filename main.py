from utils.entry_point import EntryPoint
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import (
    stream_routes
)
from settings.swagger_doc import (
    API_TITLE,
    API_DESCRIPTION,
    API_SUMMARY
)

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    summary=API_SUMMARY
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Routers
app.router.include_router(stream_routes.router)

if __name__ == "__main__":
    main_entrypoint = EntryPoint()
    main_entrypoint.run()

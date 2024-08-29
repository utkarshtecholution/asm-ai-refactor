"""
All the settings for the application will be configured here.
"""

# Uvicorn server config
UVICORN_APP = "main:app"
UVICORN_HOST = "0.0.0.0"
UVICORN_PORT = 8501
UVICORN_RELOAD =  True
UVICORN_LOG_LEVEL = "info"
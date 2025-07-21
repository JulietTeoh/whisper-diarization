import os
import uuid
import logging
from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from config import settings
from models import ModelList, ModelInfo
from services.transcription import TranscriptionService
from utils.temp_manager import TempManager
from utils.model_manager import model_manager
from exceptions import (
    AudioValidationError, 
    UnsupportedFormatError, 
    TranscriptionError
)

# Configure logging
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize temp manager
    temp_manager = TempManager(base_temp_dir=settings.temp_dir)
    app.state.temp_manager = temp_manager

    # Initialize model manager (loads models on demand)
    logger.info("Initializing model manager...")
    app.state.model_manager = model_manager
    logger.info("Model manager initialized.")

    # The TranscriptionService will be created on-demand per request
    yield

    # Cleanup on shutdown
    temp_manager.cleanup_all()
    model_manager.cleanup()


app = FastAPI(
    title="Whisper Diarization Server",
    description="OpenAI-compatible API for speech transcription with speaker diarization",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    return ModelList(
        object="list",
        data=[
            ModelInfo(
                id="whisper-1",
                created=1677532384,
                owned_by="openai-internal"
            )
        ]
    )


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    chunking_strategy: Optional[str] = Form(None),
    timestamp_granularities: Optional[List[str]] = Form(None),
    enable_vocal_separation: Optional[bool] = Form(None),
    enable_punctuation: Optional[bool] = Form(None)
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    if model not in ["whisper-1"]:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model}")

    if response_format not in ["json", "text", "srt", "verbose_json", "vtt"]:
        raise HTTPException(status_code=400, detail=f"Unsupported response format: {response_format}")

    temp_manager = app.state.temp_manager
    request_id = str(uuid.uuid4())
    temp_dir = None

    try:
        with TranscriptionService() as transcription_service:
            temp_dir = temp_manager.create_temp_dir(request_id)

            file_extension = os.path.splitext(file.filename)[1].lower()
            temp_file_path = os.path.join(temp_dir, f"input{file_extension}")

            with open(temp_file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

            logger.info(f"Processing transcription request {request_id} for file: {file.filename}")

            result = await transcription_service.transcribe(
                audio_file_path=temp_file_path,
                language=language,
                prompt=prompt,
                temperature=temperature,
                response_format=response_format,
                timestamp_granularities=timestamp_granularities,
                chunking_strategy=chunking_strategy,
                enable_vocal_separation=enable_vocal_separation,
                enable_punctuation=enable_punctuation
            )

            if response_format == "text":
                return PlainTextResponse(content=result)
            elif response_format in ["srt", "vtt"]:
                return PlainTextResponse(content=result, media_type="text/plain")
            else:
                return result

    except UnsupportedFormatError as e:
        logger.error(f"Unsupported format error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except AudioValidationError as e:
        logger.error(f"Audio validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except TranscriptionError as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        if temp_dir:
            temp_manager.cleanup_temp_dir(request_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port, log_level=settings.log_level.lower())
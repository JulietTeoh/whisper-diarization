from typing import List, Optional
from pydantic import BaseModel, Field


class TranscriptionSegment(BaseModel):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    speaker: Optional[str] = None


class TranscriptionWord(BaseModel):
    word: str
    start: float
    end: float
    speaker: Optional[str] = None


class UsageInfo(BaseModel):
    type: str = "duration"
    seconds: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class TranscriptionResponse(BaseModel):
    text: str
    task: str = "transcribe"
    language: Optional[str] = None
    duration: Optional[float] = None
    segments: Optional[List[TranscriptionSegment]] = None
    words: Optional[List[TranscriptionWord]] = None
    usage: Optional[UsageInfo] = None


class ErrorResponse(BaseModel):
    error: dict = Field(
        default_factory=lambda: {
            "message": "An error occurred",
            "type": "server_error",
            "code": "internal_server_error"
        }
    )


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]
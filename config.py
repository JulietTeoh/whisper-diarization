import os
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, computed_field

class Settings(BaseSettings):
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # Model configuration
    whisper_model: str = "medium.en"
    device_setting: str = Field(default="auto", alias="device") # Use "auto" as a safe default
    @computed_field
    @property
    def device(self) -> str:
        if self.device_setting == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device_setting

    @computed_field
    @property
    def compute_type(self) -> str:
        return "int8" if self.device == "cpu" else "float16"
    
    # Audio processing
    suppress_numerals: bool = False
    batch_size: int = 8
    
    # Workflow Configuration
    enable_stemming: bool = True
    enable_vocal_separation: bool = True
    enable_punctuation_restoration: bool = True
    enable_speaker_diarization: bool = True
    
    # Chunking configuration for long audio
    chunk_length_minutes: int = 10
    chunk_overlap_seconds: int = 30
    max_audio_duration_hours: int = 24
    max_concurrent_chunks: int = 4
    
    # HuggingFace configuration
    huggingface_token: Optional[str] = None
    hf_token: Optional[str] = None
    
    # Diarization configuration
    max_speakers: int = 8
    @computed_field
    @property
    def diarization_device(self) -> str:
        # Diarization can run on the same device as Whisper
        return self.device
    
    # Temporary files
    temp_dir: str = "/tmp"
    cleanup_temp_files: bool = True
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
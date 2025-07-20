import os
from typing import Optional

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class Settings(BaseSettings):
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # Model configuration
    whisper_model: str = "medium.en"
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    compute_type: str = "float16"
    
    # Audio processing
    enable_stemming: bool = True
    enable_punctuation_restoration: bool = True
    suppress_numerals: bool = False
    batch_size: int = 8
    
    # Workflow Configuration
    enable_vocal_separation: bool = True
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
    diarization_device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    
    # Temporary files
    temp_dir: str = "/tmp"
    cleanup_temp_files: bool = True
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
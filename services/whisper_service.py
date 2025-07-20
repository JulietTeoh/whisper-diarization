import logging
from typing import List, Optional, Tuple, Dict, Any

import faster_whisper
import torch
import numpy as np

from config import settings
from helpers import find_numeral_symbol_tokens, process_language_arg
from utils.model_manager import model_manager

logger = logging.getLogger(__name__)


class WhisperService:
    def __init__(self, use_shared_models: bool = True):
        self.use_shared_models = use_shared_models
        self.model = None
        self.pipeline = None
        
        if not use_shared_models:
            # Legacy mode: create own model instance
            self.device = self._resolve_device(settings.device)
            self.model_name = settings.whisper_model
            self.compute_type = "int8" if self.device == "cpu" else "float16"
            self._initialize_model()
        else:
            # Shared mode: use ModelManager instances
            self.model = model_manager.get_whisper_model()
            self.pipeline = model_manager.get_whisper_pipeline()
            self.device = model_manager.get_whisper_device()
            self.model_name = settings.whisper_model
            self.compute_type = model_manager.get_compute_type()
            logger.info(f"Using shared Whisper model: {self.model_name} on {self.device}")

    def _resolve_device(self, device_setting):
        if device_setting == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device_setting
    
    def _initialize_model(self):
        try:
            self.model = faster_whisper.WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type
            )
            self.pipeline = faster_whisper.BatchedInferencePipeline(self.model)
            logger.info(f"Initialized Whisper model: {self.model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize Whisper model: {e}")
            raise
    
    def transcribe_audio(
        self,
        audio_file_path: str,
        language: Optional[str] = None,
        batch_size: int = None,
        suppress_numerals: bool = False,
        temperature: float = 0.0
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        try:
            if batch_size is None:
                batch_size = settings.batch_size
            
            processed_language = process_language_arg(language, self.model_name)
            
            audio_waveform = faster_whisper.decode_audio(audio_file_path)
            
            suppress_tokens = (
                find_numeral_symbol_tokens(self.model.hf_tokenizer)
                if suppress_numerals
                else [-1]
            )
            
            if batch_size > 0:
                segments, info = self.pipeline.transcribe(
                    audio_waveform,
                    processed_language,
                    suppress_tokens=suppress_tokens,
                    batch_size=batch_size,
                    temperature=temperature
                )
            else:
                segments, info = self.model.transcribe(
                    audio_waveform,
                    processed_language,
                    suppress_tokens=suppress_tokens,
                    vad_filter=True,
                    temperature=temperature
                )
            
            segment_list = []
            full_transcript = ""
            
            for segment in segments:
                segment_dict = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "tokens": segment.tokens if hasattr(segment, 'tokens') else [],
                    "temperature": segment.temperature if hasattr(segment, 'temperature') else temperature,
                    "avg_logprob": segment.avg_logprob if hasattr(segment, 'avg_logprob') else 0.0,
                    "compression_ratio": segment.compression_ratio if hasattr(segment, 'compression_ratio') else 1.0,
                    "no_speech_prob": segment.no_speech_prob if hasattr(segment, 'no_speech_prob') else 0.0,
                    "seek": segment.seek if hasattr(segment, 'seek') else 0
                }
                segment_list.append(segment_dict)
                full_transcript += segment.text
            
            info_dict = {
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "duration_after_vad": info.duration_after_vad if hasattr(info, 'duration_after_vad') else info.duration
            }
            
            logger.info(f"Transcribed audio: {len(segment_list)} segments, language: {info.language}")
            
            return full_transcript, segment_list, info_dict
            
        except Exception as e:
            logger.error(f"Failed to transcribe audio: {e}")
            raise
    
    def transcribe_audio_data(
        self,
        audio_data: np.ndarray,
        language: Optional[str] = None,
        batch_size: int = None,
        suppress_numerals: bool = False,
        temperature: float = 0.0
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        try:
            if batch_size is None:
                batch_size = settings.batch_size
            
            processed_language = process_language_arg(language, self.model_name)
            
            suppress_tokens = (
                find_numeral_symbol_tokens(self.model.hf_tokenizer)
                if suppress_numerals
                else [-1]
            )
            
            if batch_size > 0:
                segments, info = self.pipeline.transcribe(
                    audio_data,
                    processed_language,
                    suppress_tokens=suppress_tokens,
                    batch_size=batch_size,
                    temperature=temperature
                )
            else:
                segments, info = self.model.transcribe(
                    audio_data,
                    processed_language,
                    suppress_tokens=suppress_tokens,
                    vad_filter=True,
                    temperature=temperature
                )
            
            segment_list = []
            full_transcript = ""
            
            for segment in segments:
                segment_dict = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "tokens": segment.tokens if hasattr(segment, 'tokens') else [],
                    "temperature": segment.temperature if hasattr(segment, 'temperature') else temperature,
                    "avg_logprob": segment.avg_logprob if hasattr(segment, 'avg_logprob') else 0.0,
                    "compression_ratio": segment.compression_ratio if hasattr(segment, 'compression_ratio') else 1.0,
                    "no_speech_prob": segment.no_speech_prob if hasattr(segment, 'no_speech_prob') else 0.0,
                    "seek": segment.seek if hasattr(segment, 'seek') else 0
                }
                segment_list.append(segment_dict)
                full_transcript += segment.text
            
            info_dict = {
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "duration_after_vad": info.duration_after_vad if hasattr(info, 'duration_after_vad') else info.duration
            }
            
            return full_transcript, segment_list, info_dict
            
        except Exception as e:
            logger.error(f"Failed to transcribe audio data: {e}")
            raise
    
    def cleanup(self):
        try:
            if not self.use_shared_models:
                # Only cleanup if using own model instance
                if self.model:
                    del self.model
                    self.model = None
                if self.pipeline:
                    del self.pipeline
                    self.pipeline = None
                torch.cuda.empty_cache()
                logger.info("Cleaned up Whisper model")
            else:
                # When using shared models, just clear references
                self.model = None
                self.pipeline = None
                logger.info("Cleared references to shared Whisper model")
        except Exception as e:
            logger.error(f"Failed to cleanup Whisper model: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False
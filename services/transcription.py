import os
import logging
from typing import Optional, Dict, Any, List

from services.audio_service import AudioService
from services.whisper_service import WhisperService
from services.diarization_service import DiarizationService
from services.chunk_processor import ChunkProcessor
from utils.response_formatter import ResponseFormatter
from utils.model_manager import model_manager
from config import settings

logger = logging.getLogger(__name__)


class TranscriptionService:
    def __init__(self):
        self.audio_service = AudioService()
        self.whisper_service = None
        self.diarization_service = None
        self.chunk_processor = None
        self.response_formatter = ResponseFormatter()
    
    def __enter__(self):
        # Initialize services that use shared models from ModelManager
        self.whisper_service = WhisperService(use_shared_models=True).__enter__()

        # Only initialize diarization service if enabled
        if settings.enable_speaker_diarization:
            self.diarization_service = DiarizationService(use_shared_models=True)

        self.chunk_processor = ChunkProcessor(use_shared_models=True).__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Services using shared models don't need cleanup (models are managed by ModelManager)
        if self.whisper_service:
            self.whisper_service.__exit__(exc_type, exc_val, exc_tb)
        if self.chunk_processor:
            self.chunk_processor.__exit__(exc_type, exc_val, exc_tb)
        # Note: No cleanup needed for diarization_service since it uses shared models
        return False
    
    async def transcribe(
        self,
        audio_file_path: str,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        temperature: float = 0.0,
        response_format: str = "json",
        timestamp_granularities: Optional[List[str]] = None,
        chunking_strategy: Optional[str] = None,
        enable_vocal_separation: Optional[bool] = None,
        enable_punctuation: Optional[bool] = None
    ) -> Any:
        try:
            logger.info(f"Starting transcription for: {audio_file_path}")
            
            if not self.audio_service.is_supported_format(audio_file_path):
                raise ValueError(f"Unsupported audio format: {os.path.splitext(audio_file_path)[1]}")
            
            if not self.audio_service.validate_audio_file(audio_file_path):
                raise ValueError("Audio file validation failed")
            
            temp_dir = os.path.dirname(audio_file_path)
            
            audio_info = self.audio_service.prepare_audio(audio_file_path, temp_dir, chunking_strategy, enable_vocal_separation)
            
            if audio_info["needs_chunking"]:
                logger.info("Processing long audio file with chunking")
                result = await self._process_chunked_audio(
                    audio_info, temp_dir, language, temperature, enable_punctuation
                )
            else:
                logger.info("Processing short audio file without chunking")
                result = await self._process_single_audio(
                    audio_info, temp_dir, language, temperature, enable_punctuation
                )
            
            formatted_response = self.response_formatter.format_transcription_response(
                text=result["text"],
                language=result.get("language"),
                duration=result.get("duration"),
                segments=result.get("sentences", []),
                words=result.get("words", []),
                response_format=response_format,
                timestamp_granularities=timestamp_granularities
            )
            
            logger.info(f"Transcription completed successfully")
            return formatted_response
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    async def _process_single_audio(
        self,
        audio_info: Dict[str, Any],
        temp_dir: str,
        language: Optional[str] = None,
        temperature: float = 0.0,
        enable_punctuation: Optional[bool] = None
    ) -> Dict[str, Any]:
        try:
            full_transcript, whisper_segments, info = self.whisper_service.transcribe_audio(
                audio_info["audio_file_path"],
                language=language,
                temperature=temperature,
                batch_size=settings.batch_size,
                suppress_numerals=settings.suppress_numerals
            )

            if not full_transcript.strip():
                logger.warning("Empty transcript from Whisper")
                return {
                    "text": "",
                    "segments": [],
                    "words": [],
                    "sentences": [],
                    "language": info.get("language", "en"),
                    "duration": audio_info["duration"]
                }

            # Perform speaker diarization if enabled
            if settings.enable_speaker_diarization:
                logger.info("Performing speaker diarization")
                if self.diarization_service is None:
                    self.diarization_service = DiarizationService(use_shared_models=True)
                
                diarization_result = self.diarization_service.perform_diarization(
                    temp_dir,
                    full_transcript,
                    info.get("language", "en"),
                    settings.batch_size,
                    enable_punctuation
                )

                return {
                    "text": full_transcript,
                    "segments": whisper_segments,
                    "words": diarization_result["word_speaker_mapping"],
                    "sentences": diarization_result["sentence_speaker_mapping"],
                    "language": info.get("language", "en"),
                    "duration": audio_info["duration"]
                }
            else:
                logger.info("Speaker diarization disabled, returning basic transcript")
                # Return basic transcript without speaker information
                return {
                    "text": full_transcript,
                    "segments": whisper_segments,
                    "words": [],
                    "sentences": [],
                    "language": info.get("language", "en"),
                    "duration": audio_info["duration"]
                }

        except Exception as e:
            logger.error(f"Failed to process single audio: {e}")
            raise
    
    async def _process_chunked_audio(
        self,
        audio_info: Dict[str, Any],
        temp_dir: str,
        language: Optional[str] = None,
        temperature: float = 0.0,
        enable_punctuation: Optional[bool] = None
    ) -> Dict[str, Any]:
        try:
            chunks = audio_info["chunks"]
            logger.info(f"Processing {len(chunks)} chunks")
            
            chunk_results = await self.chunk_processor.process_chunks(
                chunks,
                temp_dir,
                language=language,
                temperature=temperature,
                batch_size=settings.batch_size,
                enable_punctuation=enable_punctuation
            )
            
            merged_result = self.chunk_processor.merge_chunk_results(chunk_results)
            
            return merged_result
            
        except Exception as e:
            logger.error(f"Failed to process chunked audio: {e}")
            raise
    

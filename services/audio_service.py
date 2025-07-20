import os
import logging
from typing import List, Dict, Any, Optional
import subprocess
import librosa

from config import settings
from utils.audio_utils import AudioProcessor, AudioChunk

logger = logging.getLogger(__name__)


class AudioService:
    def __init__(self):
        self.audio_processor = AudioProcessor(
            chunk_length_minutes=settings.chunk_length_minutes,
            overlap_seconds=settings.chunk_overlap_seconds
        )
        self.stemming_enabled = settings.enable_stemming
        self.vocal_separation_enabled = settings.enable_vocal_separation
        self.target_sample_rate = 16000
        self.device = self._resolve_device(settings.device)
    
    def _resolve_device(self, device_setting):
        if device_setting == "auto":
            import torch
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device_setting
    
    def prepare_audio(self, input_file_path: str, temp_dir: str, chunking_strategy: Optional[str] = None, enable_vocal_separation: Optional[bool] = None) -> Dict[str, Any]:
        try:
            logger.info(f"Preparing audio file: {input_file_path}")
            
            if chunking_strategy and chunking_strategy != "auto":
                logger.warning(f"Unsupported chunking strategy: {chunking_strategy}. Using 'auto'.")

            vocal_target = self._extract_vocals(input_file_path, temp_dir, enable_vocal_separation)

            mono_file_path = os.path.join(temp_dir, "mono_file.wav")
            self.audio_processor.convert_to_mono_wav(vocal_target, mono_file_path)

            audio_data, sample_rate = self.audio_processor.load_audio(mono_file_path)
            duration = self.audio_processor.get_audio_duration(audio_data, sample_rate)

            needs_chunking = self.audio_processor.needs_chunking(audio_data, sample_rate)
            result = {
                "audio_file_path": mono_file_path,
                "original_file_path": input_file_path,
                "vocal_file_path": vocal_target,
                "audio_data": audio_data,
                "sample_rate": sample_rate,
                "duration": duration,
                "needs_chunking": needs_chunking,
                "chunks": []
            }
            
            if needs_chunking:
                logger.info(f'Starting chunking')
                chunks = self.audio_processor.chunk_audio(audio_data, sample_rate)
                result["chunks"] = chunks
                logger.info(f"Created {len(chunks)} chunks for processing")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to prepare audio: {e}")
            raise
    
    def _extract_vocals(self, input_file_path: str, temp_dir: str, enable_vocal_separation: Optional[bool] = None) -> str:
        # Use override parameter if provided, otherwise fall back to global config setting
        use_vocal_separation = enable_vocal_separation if enable_vocal_separation is not None else self.vocal_separation_enabled
        
        if not use_vocal_separation:
            logger.info("Vocal separation disabled, using original audio file")
            return input_file_path
        
        try:
            logger.info("Extracting vocals using Demucs")
            
            command = [
                "python", "-m", "demucs.separate",
                "-n", "htdemucs",
                "--two-stems=vocals",
                input_file_path,
                "-o", temp_dir,
                "--device", self.device
            ]
            
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode != 0:
                logger.warning(
                    f"Source splitting failed: {result.stderr}. "
                    "Using original audio file."
                )
                return input_file_path
            
            base_name = os.path.splitext(os.path.basename(input_file_path))[0]
            vocal_file_path = os.path.join(
                temp_dir, "htdemucs", base_name, "vocals.wav"
            )
            
            if os.path.exists(vocal_file_path):
                logger.info(f"Successfully extracted vocals: {vocal_file_path}")
                return vocal_file_path
            else:
                logger.warning("Vocal extraction output file not found. Using original.")
                return input_file_path
                
        except subprocess.TimeoutExpired:
            logger.error("Vocal extraction timed out. Using original audio file.")
            return input_file_path
        except Exception as e:
            logger.error(f"Failed to extract vocals: {e}. Using original audio file.")
            return input_file_path
    
    def save_audio_chunk(self, chunk: AudioChunk, temp_dir: str) -> str:
        chunk_file_path = os.path.join(temp_dir, f"chunk_{chunk.chunk_id}.wav")
        return self.audio_processor.save_chunk(chunk, chunk_file_path)
    
    def validate_audio_file(self, file_path: str) -> bool:
        try:
            audio_info = librosa.get_samplerate(file_path)
            duration = librosa.get_duration(filename=file_path)
            
            if duration > settings.max_audio_duration_hours * 3600:
                logger.error(f"Audio file too long: {duration}s > {settings.max_audio_duration_hours}h")
                return False
            
            logger.info(f"Audio validation passed: {duration:.2f}s at {audio_info}Hz")
            return True
            
        except Exception as e:
            logger.error(f"Audio validation failed: {e}")
            return False
    
    def get_supported_formats(self) -> List[str]:
        return [
            ".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", 
            ".wma", ".mp4", ".mpeg", ".mpga", ".webm"
        ]
    
    def is_supported_format(self, file_path: str) -> bool:
        file_extension = os.path.splitext(file_path)[1].lower()
        return file_extension in self.get_supported_formats()
    
    def merge_chunk_results(self, chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.audio_processor.merge_transcription_results(chunk_results)
import os
import logging
from typing import List, Dict, Any, Optional
import concurrent.futures
import asyncio
from dataclasses import dataclass

from config import settings

from utils.audio_utils import AudioChunk
from services.whisper_service import WhisperService
from services.diarization_service import DiarizationService
from utils.model_manager import model_manager

logger = logging.getLogger(__name__)


@dataclass
class ChunkResult:
    chunk_id: int
    transcript: str
    segments: List[Dict[str, Any]]
    word_speaker_mapping: List[Dict[str, Any]]
    sentence_speaker_mapping: List[Dict[str, Any]]
    start_time: float
    end_time: float
    language: str
    duration: float


class ChunkProcessor:
    def __init__(self, use_shared_models: bool = True):
        self.use_shared_models = use_shared_models
        self.whisper_service = None
        self.diarization_service = None
    
    def __enter__(self):
        # Use shared models from ModelManager instead of creating new instances
        if self.use_shared_models:
            self.whisper_service = WhisperService(use_shared_models=True)
            
            # Only initialize diarization service if enabled
            if settings.enable_speaker_diarization:
                self.diarization_service = DiarizationService(use_shared_models=True)
        else:
            # Legacy mode: create own instances
            self.whisper_service = WhisperService(use_shared_models=False)
            
            # Only initialize diarization service if enabled
            if settings.enable_speaker_diarization:
                self.diarization_service = DiarizationService(use_shared_models=False)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.whisper_service:
            self.whisper_service.cleanup()
        if self.diarization_service and not self.use_shared_models:
            # Only cleanup diarization service if not using shared models
            self.diarization_service.cleanup()
        return False
    
    async def process_chunks(
        self,
        chunks: List[AudioChunk],
        temp_dir: str,
        language: Optional[str] = None,
        temperature: float = 0.0,
        batch_size: int = None,
        enable_punctuation: Optional[bool] = None
    ) -> List[ChunkResult]:
        try:
            logger.info(f"Processing {len(chunks)} audio chunks with max {settings.max_concurrent_chunks} concurrent")
            
            # Create semaphore to limit concurrent processing
            semaphore = asyncio.Semaphore(settings.max_concurrent_chunks)
            
            # Create tasks for parallel processing
            tasks = []
            for i, chunk in enumerate(chunks):
                task = self._process_chunk_with_semaphore(
                    semaphore, chunk, i, len(chunks), temp_dir, language, temperature, batch_size, enable_punctuation
                )
                tasks.append(task)
            
            # Process chunks concurrently while maintaining order
            chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check for exceptions in results
            for i, result in enumerate(chunk_results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to process chunk {i+1}: {result}")
                    raise result
            
            logger.info(f"Completed processing {len(chunk_results)} chunks")
            return chunk_results
            
        except Exception as e:
            logger.error(f"Failed to process chunks: {e}")
            raise
    
    async def _process_chunk_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        chunk: AudioChunk,
        chunk_index: int,
        total_chunks: int,
        temp_dir: str,
        language: Optional[str] = None,
        temperature: float = 0.0,
        batch_size: int = None,
        enable_punctuation: Optional[bool] = None
    ) -> ChunkResult:
        async with semaphore:
            logger.info(f"Processing chunk {chunk_index+1}/{total_chunks}")
            return await self._process_single_chunk(
                chunk, temp_dir, language, temperature, batch_size, enable_punctuation
            )
    
    async def _process_single_chunk(
        self,
        chunk: AudioChunk,
        temp_dir: str,
        language: Optional[str] = None,
        temperature: float = 0.0,
        batch_size: int = None,
        enable_punctuation: Optional[bool] = None
    ) -> ChunkResult:
        try:
            chunk_temp_dir = os.path.join(temp_dir, f"chunk_{chunk.chunk_id}")
            os.makedirs(chunk_temp_dir, exist_ok=True)
            
            chunk_file_path = os.path.join(chunk_temp_dir, f"chunk_{chunk.chunk_id}.wav")
            
            import soundfile as sf
            sf.write(chunk_file_path, chunk.audio_data, chunk.sample_rate)
            
            full_transcript, whisper_segments, info = self.whisper_service.transcribe_audio_data(
                chunk.audio_data,
                language=language,
                temperature=temperature,
                batch_size=batch_size
            )
            
            if not full_transcript.strip():
                logger.warning(f"Empty transcript for chunk {chunk.chunk_id}")
                return ChunkResult(
                    chunk_id=chunk.chunk_id,
                    transcript="",
                    segments=[],
                    word_speaker_mapping=[],
                    sentence_speaker_mapping=[],
                    start_time=chunk.start_time,
                    end_time=chunk.end_time,
                    language=info.get("language", "en"),
                    duration=chunk.end_time - chunk.start_time
                )
            
            # Perform speaker diarization if enabled
            if settings.enable_speaker_diarization:
                diarization_result = self.diarization_service.perform_diarization(
                    chunk_temp_dir,
                    full_transcript,
                    info.get("language", "en"),
                    batch_size,
                    enable_punctuation
                )
                
                adjusted_word_mapping = self._adjust_word_timestamps(
                    diarization_result["word_speaker_mapping"], chunk.start_time
                )
                adjusted_sentence_mapping = self._adjust_sentence_timestamps(
                    diarization_result["sentence_speaker_mapping"], chunk.start_time
                )
            else:
                # Skip diarization, return empty mappings
                adjusted_word_mapping = []
                adjusted_sentence_mapping = []
            
            adjusted_segments = self._adjust_timestamps(
                whisper_segments, chunk.start_time
            )
            
            return ChunkResult(
                chunk_id=chunk.chunk_id,
                transcript=full_transcript,
                segments=adjusted_segments,
                word_speaker_mapping=adjusted_word_mapping,
                sentence_speaker_mapping=adjusted_sentence_mapping,
                start_time=chunk.start_time,
                end_time=chunk.end_time,
                language=info.get("language", "en"),
                duration=chunk.end_time - chunk.start_time
            )
            
        except Exception as e:
            logger.error(f"Failed to process chunk {chunk.chunk_id}: {e}")
            raise
    
    def _adjust_timestamps(self, segments: List[Dict[str, Any]], offset: float) -> List[Dict[str, Any]]:
        adjusted_segments = []
        for segment in segments:
            adjusted_segment = segment.copy()
            adjusted_segment["start"] += offset
            adjusted_segment["end"] += offset
            adjusted_segments.append(adjusted_segment)
        return adjusted_segments
    
    def _adjust_word_timestamps(self, word_mapping: List[Dict[str, Any]], offset: float) -> List[Dict[str, Any]]:
        adjusted_words = []
        for word in word_mapping:
            adjusted_word = word.copy()
            adjusted_word["start_time"] += offset * 1000  # Convert to milliseconds
            adjusted_word["end_time"] += offset * 1000
            adjusted_words.append(adjusted_word)
        return adjusted_words
    
    def _adjust_sentence_timestamps(self, sentence_mapping: List[Dict[str, Any]], offset: float) -> List[Dict[str, Any]]:
        adjusted_sentences = []
        for sentence in sentence_mapping:
            adjusted_sentence = sentence.copy()
            adjusted_sentence["start_time"] += offset * 1000  # Convert to milliseconds
            adjusted_sentence["end_time"] += offset * 1000
            adjusted_sentences.append(adjusted_sentence)
        return adjusted_sentences
    
    def merge_chunk_results(self, chunk_results: List[ChunkResult]) -> Dict[str, Any]:
        try:
            if not chunk_results:
                return {
                    "text": "",
                    "segments": [],
                    "words": [],
                    "sentences": [],
                    "language": "en",
                    "duration": 0.0
                }
            
            merged_text = []
            merged_segments = []
            merged_words = []
            merged_sentences = []
            
            total_duration = 0.0
            primary_language = chunk_results[0].language
            
            for result in chunk_results:
                if result.transcript.strip():
                    merged_text.append(result.transcript.strip())
                
                merged_segments.extend(result.segments)
                merged_words.extend(result.word_speaker_mapping)
                merged_sentences.extend(result.sentence_speaker_mapping)
                
                total_duration = max(total_duration, result.end_time)
            
            merged_words = self._resolve_speaker_conflicts(merged_words)
            merged_sentences = self._resolve_sentence_conflicts(merged_sentences)
            
            return {
                "text": " ".join(merged_text),
                "segments": merged_segments,
                "words": merged_words,
                "sentences": merged_sentences,
                "language": primary_language,
                "duration": total_duration
            }
            
        except Exception as e:
            logger.error(f"Failed to merge chunk results: {e}")
            raise
    
    def _resolve_speaker_conflicts(self, words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not words:
            return words
        
        words.sort(key=lambda x: x.get("start_time", 0))
        
        resolved_words = []
        speaker_mapping = {}
        
        for word in words:
            start_time = word.get("start_time", 0)
            end_time = word.get("end_time", 0)
            speaker = word.get("speaker", "Speaker 0")
            
            if start_time not in speaker_mapping:
                speaker_mapping[start_time] = speaker
            
            resolved_word = word.copy()
            resolved_word["speaker"] = speaker_mapping[start_time]
            resolved_words.append(resolved_word)
        
        return resolved_words
    
    def _resolve_sentence_conflicts(self, sentences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not sentences:
            return sentences
        
        sentences.sort(key=lambda x: x.get("start_time", 0))
        
        resolved_sentences = []
        
        for sentence in sentences:
            resolved_sentence = sentence.copy()
            resolved_sentences.append(resolved_sentence)
        
        return resolved_sentences
    

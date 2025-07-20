import json
import os
import logging
from typing import List, Dict, Any, Optional
import torch
import torchaudio
import numpy as np

from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    postprocess_results,
    preprocess_text,
)
from deepmultilingualpunctuation import PunctuationModel

from config import settings
from utils.model_manager import model_manager
from helpers import (
    create_config,
    get_words_speaker_mapping,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    langs_to_iso,
    punct_model_langs,
)

logger = logging.getLogger(__name__)


class DiarizationService:
    def __init__(self, use_shared_models: bool = True):
        self.use_shared_models = use_shared_models
        self.alignment_model = None
        self.alignment_tokenizer = None
        self.punct_model = None

        if not use_shared_models:
            # Legacy mode: create own model instances
            self.device = self._resolve_device(settings.diarization_device)
            self._initialize_models()
        else:
            # Shared mode: use ModelManager instances
            self.device = model_manager.get_diarization_device()
            if settings.enable_speaker_diarization:
                self.alignment_model, self.alignment_tokenizer = model_manager.get_alignment_model()
            if settings.enable_punctuation_restoration:
                self.punct_model = model_manager.get_punctuation_model()
            logger.info(f"Using shared diarization models on {self.device}")
    
    def _resolve_device(self, device_setting):
        if device_setting == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device_setting
    
    def _initialize_models(self):
        try:
            self.alignment_model, self.alignment_tokenizer = load_alignment_model(
                self.device,
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            logger.info(f"Initialized alignment model on {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize alignment model: {e}")
            raise

    def perform_diarization(
        self,
        temp_dir: str,
        full_transcript: str,
        language: str,
        batch_size: int = None,
        enable_punctuation: Optional[bool] = None
    ) -> Dict[str, Any]:
        try:
            if batch_size is None:
                batch_size = settings.batch_size

            # Use the existing mono_file.wav created by AudioService - no need to reload/resave
            mono_file_path = os.path.join(temp_dir, "mono_file.wav")

            # Verify the file exists (it should have been created by AudioService)
            if not os.path.exists(mono_file_path):
                raise FileNotFoundError(f"mono_file.wav not found at {mono_file_path}. AudioService should create this first.")

            # Load the audio for word alignment (we still need the waveform for that)
            audio_waveform, sample_rate = torchaudio.load(mono_file_path)
            audio_waveform = audio_waveform.squeeze().numpy()

            word_timestamps = self._perform_alignment(
                audio_waveform, full_transcript, language, batch_size
            )

            speaker_timestamps = self._perform_speaker_diarization(temp_dir)

            # Handle case where diarization failed (empty speaker_timestamps)
            if not speaker_timestamps:
                logger.warning("No speaker timestamps found. Assigning all words to Speaker 0.")
                # Create a fallback: assign all audio to a single speaker
                if word_timestamps:
                    # Get the full duration from first to last word
                    first_word_start = min(w["start"] for w in word_timestamps) * 1000
                    last_word_end = max(w["end"] for w in word_timestamps) * 1000
                    speaker_timestamps = [[int(first_word_start), int(last_word_end), 0]]
                else:
                    # No words either, create a minimal speaker timestamp
                    speaker_timestamps = [[0, 1000, 0]]  # 1 second, Speaker 0

            word_speaker_mapping = get_words_speaker_mapping(
                word_timestamps, speaker_timestamps, "start"
            )

            # Apply punctuation restoration if enabled
            use_punctuation = enable_punctuation if enable_punctuation is not None else settings.enable_punctuation_restoration
            if use_punctuation:
                word_speaker_mapping = self._apply_punctuation_restoration(
                    word_speaker_mapping, language
                )

            word_speaker_mapping = get_realigned_ws_mapping_with_punctuation(
                word_speaker_mapping
            )

            sentence_speaker_mapping = get_sentences_speaker_mapping(
                word_speaker_mapping, speaker_timestamps
            )

            logger.info(f"Diarization completed: {len(sentence_speaker_mapping)} sentences")

            return {
                "word_speaker_mapping": word_speaker_mapping,
                "sentence_speaker_mapping": sentence_speaker_mapping,
                "speaker_timestamps": speaker_timestamps,
                "word_timestamps": word_timestamps
            }

        except Exception as e:
            logger.error(f"Failed to perform diarization: {e}")
            raise
    
    def _perform_alignment(
        self,
        audio_waveform: np.ndarray,
        full_transcript: str,
        language: str,
        batch_size: int
    ) -> List[Dict[str, Any]]:
        try:
            emissions, stride = generate_emissions(
                self.alignment_model,
                torch.from_numpy(audio_waveform)
                .to(self.alignment_model.dtype)
                .to(self.alignment_model.device),
                batch_size=batch_size,
            )

            tokens_starred, text_starred = preprocess_text(
                full_transcript,
                romanize=True,
                language=langs_to_iso[language],
            )

            segments, scores, blank_token = get_alignments(
                emissions,
                tokens_starred,
                self.alignment_tokenizer,
            )

            spans = get_spans(tokens_starred, segments, blank_token)

            word_timestamps = postprocess_results(text_starred, spans, stride, scores)

            return word_timestamps

        except Exception as e:
            logger.error(f"Failed to perform alignment: {e}")
            raise
    
    def _perform_speaker_diarization(self, temp_dir: str) -> List[List[int]]:
        try:
            if self.use_shared_models:
                # Use shared NeMo model from ModelManager with proper __call__ method
                msdd_model = model_manager.get_nemo_diarizer()

                # Find the audio file in temp_dir (should be mono_file.wav)
                audio_file_path = os.path.join(temp_dir, "mono_file.wav")
                
                # Verify the audio file exists and is readable
                if not os.path.exists(audio_file_path):
                    raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

                # Check audio file properties for debugging
                try:
                    import librosa
                    duration = librosa.get_duration(path=audio_file_path)
                    logger.info(f"Audio file {audio_file_path} duration: {duration}s")
                    if duration <= 0:
                        raise ValueError(f"Audio file has zero or negative duration: {duration}s")
                except Exception as e:
                    logger.warning(f"Could not verify audio file properties: {e}")

                # Use the official __call__ method which handles all config updates properly
                # This replaces manual config updates and direct diarize() calls
                msdd_model(
                    audio_filepath=audio_file_path,
                    out_dir=temp_dir,
                    batch_size=settings.batch_size if hasattr(settings, 'batch_size') else 64,
                    num_workers=1,  # Keep conservative for server usage
                    max_speakers=10,  # Add reasonable default for max speakers
                    num_speakers=None,  # Let NeMo auto-detect number of speakers
                    verbose=True    # Print Errors
                )

            else:
                # Legacy mode: create new model instance (for backward compatibility)
                msdd_model = NeuralDiarizer(cfg=create_config(temp_dir)).to(self.device)
                msdd_model.diarize()
                
                del msdd_model
                torch.cuda.empty_cache()

            # Parse the results (same for both modes)
            speaker_timestamps = []
            rttm_file_path = os.path.join(temp_dir, "pred_rttms", "mono_file.rttm")
            
            # Check if RTTM file exists (might not if NeMo detected only silence)
            if os.path.exists(rttm_file_path):
                with open(rttm_file_path, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        line_list = line.split(" ")
                        start_ms = int(float(line_list[3]) * 1000)  # RTTM start time is field 4 (index 3)
                        duration_ms = int(float(line_list[4]) * 1000)  # Duration is field 5 (index 4)
                        end_ms = start_ms + duration_ms
                        speaker_id = int(line_list[7].split("_")[-1])  # Speaker ID is field 8 (index 7)
                        speaker_timestamps.append([start_ms, end_ms, speaker_id])
            else:
                logger.warning(f"RTTM file not found: {rttm_file_path}. NeMo may have detected only silence.")
                # Return empty speaker timestamps - this will be handled upstream
            
            return speaker_timestamps
            
        except Exception as e:
            error_msg = str(e)
            if "silence" in error_msg.lower() or "contains silence" in error_msg:
                logger.warning(f"NeMo detected silence in audio: {e}")
                # Return empty speaker timestamps instead of raising
                return []
            else:
                logger.error(f"Failed to perform speaker diarization: {e}")
                raise
    
    def _apply_punctuation_restoration(
        self, 
        word_speaker_mapping: List[Dict[str, Any]], 
        language: str
    ) -> List[Dict[str, Any]]:
        try:
            if language not in punct_model_langs:
                logger.warning(
                    f"Punctuation restoration is not available for {language} language. "
                    "Using the original punctuation."
                )
                return word_speaker_mapping
            
            if self.punct_model is None:
                if self.use_shared_models:
                    if settings.enable_punctuation_restoration:
                        self.punct_model = model_manager.get_punctuation_model()
                    else:
                        logger.warning("Punctuation restoration is disabled. Skipping.")
                        return word_speaker_mapping
                else:
                    self.punct_model = PunctuationModel(model="kredor/punctuate-all")
            
            words_list = [word_dict["word"] for word_dict in word_speaker_mapping]
            
            labeled_words = self.punct_model.predict(words_list, chunk_size=230)
            
            import re
            ending_puncts = ".?!"
            model_puncts = ".,;:!?"
            
            is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)
            
            for word_dict, labeled_tuple in zip(word_speaker_mapping, labeled_words):
                word = word_dict["word"]
                if (
                    word
                    and labeled_tuple[1] in ending_puncts
                    and (word[-1] not in model_puncts or is_acronym(word))
                ):
                    word += labeled_tuple[1]
                    if word.endswith(".."):
                        word = word.rstrip(".")
                    word_dict["word"] = word
            
            return word_speaker_mapping
            
        except Exception as e:
            logger.error(f"Failed to apply punctuation restoration: {e}")
            return word_speaker_mapping
    
    def cleanup(self):
        try:
            if not self.use_shared_models:
                # Only cleanup if using own model instances
                if self.alignment_model:
                    del self.alignment_model
                if self.alignment_tokenizer:
                    del self.alignment_tokenizer
                if self.punct_model:
                    del self.punct_model
                torch.cuda.empty_cache()
                logger.info("Cleaned up diarization models")
            else:
                # When using shared models, just clear references
                self.alignment_model = None
                self.alignment_tokenizer = None
                self.punct_model = None
                logger.info("Cleared references to shared diarization models")
        except Exception as e:
            logger.error(f"Failed to cleanup diarization models: {e}")
    
    def __del__(self):
        if not self.use_shared_models:
            self.cleanup()
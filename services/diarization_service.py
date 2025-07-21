import os
import logging
from typing import List, Dict, Any, Optional
import torch
import numpy as np

from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    postprocess_results,
    preprocess_text,
)
from whisperx.audio import load_audio

from config import settings
from utils.model_manager import model_manager
from helpers import (
    get_words_speaker_mapping,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    langs_to_iso,
    punct_model_langs,
)

logger = logging.getLogger(__name__)


class DiarizationService:
    """
    Service to perform speaker diarization, including word-level alignment and punctuation restoration.
    It retrieves all necessary models from the global ModelManager upon initialization.
    """

    def __init__(self):
        """
        Initializes the DiarizationService by fetching required models from the ModelManager.
        """
        logger.info("Initializing DiarizationService...")
        self.sample_rate = 16000 # whisperx.load_audio resamples to 16kHz
        self.device = model_manager.get_diarization_device()

        self.alignment_model = None
        self.alignment_tokenizer = None
        self.punct_model = None


        self.pyannote_pipeline = model_manager.get_pyannote_pipeline()
        if settings.enable_speaker_diarization:
            self.alignment_model, self.alignment_tokenizer = model_manager.get_alignment_model()
        if settings.enable_punctuation_restoration:
            self.punct_model = model_manager.get_punctuation_model()

        logger.info(f"DiarizationService initialized on device: {self.device}")

    def perform_diarization(
        self,
        audio_file_path: str,
        full_transcript: str,
        language: str,
        batch_size: int = None,
        enable_punctuation: Optional[bool] = None
    ) -> Dict[str, Any]:
        try:
            if batch_size is None:
                batch_size = settings.batch_size

            # Verify the file exists (it should have been created by AudioService)
            if not os.path.exists(audio_file_path):
                raise FileNotFoundError(
                    f"mono_file.wav not found at {audio_file_path}. AudioService should create this first.")

            # 1. Load audio waveform once
            # audio_waveform, sample_rate = torchaudio.load(mono_file_path)
            # audio_waveform = audio_waveform.squeeze().numpy()
            audio_waveform = load_audio(audio_file_path)

            # 2. Perform word-level alignment
            word_timestamps = self._perform_alignment(
                audio_waveform, full_transcript, language, batch_size
            )

            # 3. Perform speaker diarization
            speaker_timestamps = self._perform_speaker_diarization(audio_waveform)

            # 4. Fallback if diarization returns no speakers
            if not speaker_timestamps:
                logger.warning("No speaker timestamps found. Assigning all words to Speaker 0.")
                # Create a fallback: assign all audio to a single speaker
                if word_timestamps: # Get the full duration from first to last word
                    first_word_start = min(w["start"] for w in word_timestamps) * 1000
                    last_word_end = max(w["end"] for w in word_timestamps) * 1000
                    speaker_timestamps = [[int(first_word_start), int(last_word_end), 0]]
                else:
                    duration_ms = int(len(audio_waveform) / self.sample_rate * 1000)  # Assuming 16kHz
                    speaker_timestamps = [[0, duration_ms, 0]]  # Assign whole audio to Speaker 0

            # 5. Map words to speakers
            word_speaker_mapping = get_words_speaker_mapping(
                word_timestamps, speaker_timestamps, "start"
            )

            # 6. Apply punctuation restoration if enabled
            use_punctuation = enable_punctuation if enable_punctuation is not None else settings.enable_punctuation_restoration
            if use_punctuation:
                word_speaker_mapping = self._apply_punctuation_restoration(
                    word_speaker_mapping, language
                )

            # 7. Realign mapping and create sentences
            word_speaker_mapping = get_realigned_ws_mapping_with_punctuation(word_speaker_mapping)
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
        if not self.alignment_model or not self.alignment_tokenizer:
            logger.error("Alignment model is not available. Cannot perform alignment.")
            raise RuntimeError("Alignment models not initialized.")
        try:
            emissions, stride = generate_emissions(
                self.alignment_model,
                torch.from_numpy(audio_waveform).to(self.alignment_model.dtype).to(self.alignment_model.device),
                batch_size=batch_size,
            )
            tokens_starred, text_starred = preprocess_text(
                full_transcript, romanize=True, language=langs_to_iso[language],
            )
            segments, scores, blank_token = get_alignments(
                emissions, tokens_starred, self.alignment_tokenizer,
            )
            spans = get_spans(tokens_starred, segments, blank_token)
            word_timestamps = postprocess_results(text_starred, spans, stride, scores)
            return word_timestamps
        except Exception as e:
            logger.error(f"Failed to perform alignment: {e}")
            raise
    
    def _perform_speaker_diarization(self, audio_waveform: np.ndarray) -> List[List[int]]:
        """Runs the Pyannote pipeline on the audio waveform."""
        if not self.pyannote_pipeline:
            logger.error("Pyannote pipeline is not available. Cannot perform speaker diarization.")
            return []  # Return empty list to trigger fallback
        try:
            audio_input = {
                'waveform': torch.from_numpy(audio_waveform[None, :]),
                'sample_rate': self.sample_rate
            }

            max_speakers = getattr(settings, 'max_speakers', 8)
            diarization_result = self.pyannote_pipeline(
                audio_input, max_speakers=max_speakers
            )
            # print('=' * 50)
            # print(diarization_result)
            # print('=' * 50)

            # Robustly map speaker labels (e.g., 'SPEAKER_00') to integer IDs
            speaker_labels = sorted(diarization_result.labels())
            speaker_map = {label: i for i, label in enumerate(speaker_labels)}

            # Convert pyannote result to our expected format: [[start_ms, end_ms, speaker_id], ...]
            speaker_timestamps = []
            for segment, _, speaker_label in diarization_result.itertracks(yield_label=True):
                start_ms = int(segment.start * 1000)
                end_ms = int(segment.end * 1000)
                speaker_id = speaker_map[speaker_label]
                speaker_timestamps.append([start_ms, end_ms, speaker_id])
            
            # Sort by start time for consistency
            speaker_timestamps.sort(key=lambda x: x[0])
            logger.info(f"Pyannote diarization completed: {len(speaker_timestamps)} segments found")
            return speaker_timestamps

        except Exception as e:
            logger.error(f"Failed to perform speaker diarization with Pyannote: {e}")
            return [] # Return empty list to trigger fallback behavior upstream
    
    def _apply_punctuation_restoration(
        self, 
        word_speaker_mapping: List[Dict[str, Any]], 
        language: str
    ) -> List[Dict[str, Any]]:
        if language not in punct_model_langs:
            logger.warning(f"Punctuation restoration not available for '{language}'.")
            return word_speaker_mapping

        if not self.punct_model:
            logger.warning("Punctuation model not available, skipping restoration.")
            return word_speaker_mapping

        try:
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
        """Clears local references to shared models."""
        self.alignment_model = None
        self.alignment_tokenizer = None
        self.punct_model = None
        self.pyannote_pipeline = None
        logger.info("Cleared references to shared diarization models.")

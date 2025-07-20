import os
import logging
from typing import List, Tuple
import librosa
import soundfile as sf
import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)


class AudioChunk:
    """A data class for holding audio chunk information."""

    def __init__(self, audio_data: np.ndarray, start_time: float, end_time: float, chunk_id: int):
        self.audio_data = audio_data
        self.start_time = start_time
        self.end_time = end_time
        self.chunk_id = chunk_id
        self.sample_rate = 16000


class AudioProcessor:
    """Handles loading, chunking, and preparing audio files for processing."""

    def __init__(self, chunk_length_minutes: int = 10, overlap_seconds: int = 30):
        self.chunk_length_minutes = chunk_length_minutes
        self.overlap_seconds = overlap_seconds
        self.target_sample_rate = 16000

    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Loads an audio file into a NumPy array.

        Args:
            file_path: Path to the audio file.

        Returns:
            A tuple containing the audio data as a NumPy array and the sample rate.
        """
        try:
            audio_data, sample_rate = librosa.load(file_path, sr=self.target_sample_rate, mono=True)
            logger.info(f"Loaded audio: {len(audio_data)} samples at {sample_rate} Hz")
            return audio_data, sample_rate
        except Exception as e:
            logger.error(f"Failed to load audio file {file_path}: {e}")
            raise

    def get_audio_duration(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """Calculates the duration of an audio array in seconds."""
        return len(audio_data) / sample_rate

    def needs_chunking(self, audio_data: np.ndarray, sample_rate: int) -> bool:
        """Determines if an audio file is long enough to require chunking."""
        duration_minutes = self.get_audio_duration(audio_data, sample_rate) / 60
        return duration_minutes > self.chunk_length_minutes

    def chunk_audio(self, audio_data: np.ndarray, sample_rate: int) -> List[AudioChunk]:
        """
        Splits a long audio file into smaller, overlapping chunks.

        Args:
            audio_data: The NumPy array of the audio file.
            sample_rate: The sample rate of the audio.

        Returns:
            A list of AudioChunk objects.
        """
        duration_seconds = self.get_audio_duration(audio_data, sample_rate)
        chunk_length_seconds = self.chunk_length_minutes * 60

        if duration_seconds <= chunk_length_seconds:
            return [AudioChunk(audio_data, 0.0, duration_seconds, 0)]

        chunks = []
        chunk_id = 0

        start_time = 0.0
        while start_time < duration_seconds:
            end_time = min(start_time + chunk_length_seconds, duration_seconds)

            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)

            chunk_audio = audio_data[start_sample:end_sample]

            chunks.append(AudioChunk(
                audio_data=chunk_audio,
                start_time=start_time,
                end_time=end_time,
                chunk_id=chunk_id
            ))

            # Move the window forward, considering the overlap
            start_time += chunk_length_seconds - self.overlap_seconds
            chunk_id += 1

        logger.info(f"Created {len(chunks)} chunks for audio duration {duration_seconds:.2f}s")
        return chunks

    def save_chunk(self, chunk: AudioChunk, output_path: str) -> str:
        """Saves an audio chunk to a WAV file."""
        try:
            sf.write(output_path, chunk.audio_data, chunk.sample_rate)
            logger.debug(f"Saved chunk {chunk.chunk_id} to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to save chunk {chunk.chunk_id}: {e}")
            raise

    def convert_to_mono_wav(self, input_path: str, output_path: str) -> str:
        """
        Converts any audio file to a mono 16kHz WAV file, which is required for NeMo.
        This method is critical for ensuring compatibility with the diarization model.

        Args:
            input_path: Path to the source audio file.
            output_path: Path to save the converted WAV file.

        Returns:
            The path to the converted output file.
        """
        try:
            logger.info(f"Converting {input_path} to mono WAV at {self.target_sample_rate}Hz.")

            # --- FIX ---
            # Replaced pydub with librosa for robust audio loading and conversion.
            # librosa.load handles resampling to target_sample_rate and converting to mono.
            # This is the standard and most reliable approach for ML audio pipelines.
            audio_data, _ = librosa.load(
                input_path,
                sr=self.target_sample_rate,
                mono=True
            )

            # Librosa returns a float32 numpy array. We convert it to a torch tensor
            # and save with torchaudio to create a clean, standard WAV file that
            # NeMo's VAD module can read without errors.
            audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)

            torchaudio.save(
                output_path,
                audio_tensor,
                self.target_sample_rate,
                channels_first=True
            )
            logger.info(f"Successfully converted and saved mono WAV: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to convert audio file {input_path} to mono WAV: {e}")
            raise

    def merge_transcription_results(self, chunk_results: List[dict]) -> dict:
        """
        Merges transcription results from multiple chunks into a single result.
        (This method is a placeholder and may need more sophisticated logic for
        handling speaker labels and timestamps across chunk boundaries).
        """
        if not chunk_results:
            return {"text": "", "segments": [], "words": []}

        # This is a simplified merge. A more advanced implementation would be needed
        # to properly handle speaker mapping across chunks.
        full_text = " ".join(res.get("text", "") for res in chunk_results)
        all_segments = [seg for res in chunk_results for seg in res.get("segments", [])]
        all_words = [word for res in chunk_results for word in res.get("words", [])]

        return {
            "text": full_text.strip(),
            "segments": all_segments,
            "words": all_words
        }
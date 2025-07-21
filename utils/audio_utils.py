import logging
from typing import List, Tuple
import librosa
import soundfile as sf
import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)


from dataclasses import dataclass

@dataclass
class AudioChunk:
    """A data class for holding audio chunk information."""
    audio_data: np.ndarray
    start_time: float
    end_time: float
    chunk_id: int
    sample_rate: int = 16000


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
        try:
            logger.info(f"Converting {input_path} to mono WAV at {self.target_sample_rate}Hz.")
            audio_data, _ = librosa.load(input_path, sr=self.target_sample_rate, mono=True)
            audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
            torchaudio.save(output_path, audio_tensor, self.target_sample_rate, channels_first=True)
            logger.info(f"Successfully converted and saved mono WAV: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to convert audio file {input_path} to mono WAV: {e}")
            raise

    def merge_diarization_results(self, chunk_results: List[Tuple[dict, AudioChunk]]) -> dict:
        """
        Merges diarization results from multiple chunks.

        This function adjusts timestamps to be absolute and provides a basic framework
        for re-mapping speaker labels. For production use, the speaker re-mapping
        should be replaced with a more robust algorithm based on speaker embeddings.
        """
        if not chunk_results:
            return {"text": "", "segments": []}

        final_segments = []
        full_text = []

        # Sort chunks by start time to process them in order
        sorted_chunks = sorted(chunk_results, key=lambda x: x[1].start_time)

        # --- Placeholder for advanced speaker mapping ---
        # In a real system, you would build a global speaker map here by clustering
        # speaker embeddings from all chunks *before* processing segments.
        # For this example, we use a simpler (but less accurate) sequential mapping.
        global_speaker_map = {}
        next_global_speaker_id = 0

        for result, chunk in sorted_chunks:
            if not result or "segments" not in result:
                continue

            full_text.append(result.get("text", ""))

            # Map local speaker IDs in this chunk to global IDs
            local_to_global_id_map = {}
            for segment in sorted(result["segments"], key=lambda x: x['start']):
                local_speaker = segment.get("speaker")
                if local_speaker not in local_to_global_id_map:
                    # This naive mapping assumes new speakers in each chunk are new globally.
                    # This is where you would consult the advanced clustering result.
                    if local_speaker not in global_speaker_map:
                        global_speaker_map[local_speaker] = f"speaker_{next_global_speaker_id}"
                        next_global_speaker_id += 1
                    local_to_global_id_map[local_speaker] = global_speaker_map[local_speaker]

                # Adjust timestamps and speaker labels
                segment['start'] += chunk.start_time
                segment['end'] += chunk.start_time
                segment['speaker'] = local_to_global_id_map[local_speaker]

                final_segments.append(segment)

        # Post-processing: Merge consecutive segments from the same speaker
        if not final_segments:
            return {"text": " ".join(full_text).strip(), "segments": []}

        merged_segments = [final_segments[0]]
        for next_seg in final_segments[1:]:
            last_seg = merged_segments[-1]
            if last_seg['speaker'] == next_seg['speaker'] and last_seg['end'] >= next_seg['start']:
                last_seg['end'] = max(last_seg['end'], next_seg['end'])  # Extend the segment
                last_seg['text'] += " " + next_seg.get('text', '')
            else:
                merged_segments.append(next_seg)

        return {
            "text": " ".join(full_text).strip(),
            "segments": merged_segments
        }

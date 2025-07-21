from typing import List, Optional, Dict, Any
import json

from models import (
    TranscriptionResponse, 
    TranscriptionSegment, 
    TranscriptionWord, 
    UsageInfo
)


class ResponseFormatter:
    @staticmethod
    def format_transcription_response(
        text: str,
        language: Optional[str] = None,
        duration: Optional[float] = None,
        segments: Optional[List[Dict[str, Any]]] = None,
        words: Optional[List[Dict[str, Any]]] = None,
        response_format: str = "json",
        timestamp_granularities: Optional[List[str]] = None,
        whisper_segments: Optional[List[Dict[str, Any]]] = None,
        sentences: Optional[List[Dict[str, Any]]] = None
    ) -> Any:
        if timestamp_granularities is None:
            timestamp_granularities = []

        if sentences:
            # Check if we have sentence speaker mapping for speaker-aware text
            text = ResponseFormatter._format_speaker_aware_text(sentences)

        if response_format == "text":
            return text
        
        if response_format == "srt":
            return ResponseFormatter._format_srt(sentences or segments or [])
        
        if response_format == "vtt":
            return ResponseFormatter._format_vtt(sentences or segments or [])
        
        formatted_segments = []
        formatted_words = []
        
        # For verbose JSON, use whisper_segments (which have proper structure)
        # For regular processing, use segments (sentence speaker mapping)
        segments_to_use = whisper_segments if response_format == "verbose_json" else (sentences or segments)
        
        if segments_to_use:
            for i, segment in enumerate(segments_to_use):
                if response_format == "verbose_json":
                    # Whisper segments already have start/end in seconds, proper structure
                    formatted_segments.append(TranscriptionSegment(
                        id=i,
                        seek=segment.get("seek", 0),
                        start=segment.get("start", 0.0),
                        end=segment.get("end", 0.0),
                        text=segment.get("text", ""),
                        tokens=segment.get("tokens", []),
                        temperature=segment.get("temperature", 0.0),
                        avg_logprob=segment.get("avg_logprob", 0.0),
                        compression_ratio=segment.get("compression_ratio", 1.0),
                        no_speech_prob=segment.get("no_speech_prob", 0.0),
                        speaker=None  # Whisper segments don't have speaker info
                    ))
                else:
                    # Sentence speaker mapping has start_time/end_time in milliseconds
                    formatted_segments.append(TranscriptionSegment(
                        id=i,
                        seek=segment.get("seek", 0),
                        start=segment.get("start_time", 0) / 1000.0,
                        end=segment.get("end_time", 0) / 1000.0,
                        text=segment.get("text", ""),
                        tokens=segment.get("tokens", []),
                        temperature=segment.get("temperature", 0.0),
                        avg_logprob=segment.get("avg_logprob", 0.0),
                        compression_ratio=segment.get("compression_ratio", 1.0),
                        no_speech_prob=segment.get("no_speech_prob", 0.0),
                        speaker=segment.get("speaker", "Speaker 0")
                    ))
        
        if words:
            for word in words:
                formatted_words.append(TranscriptionWord(
                    word=word.get("word", ""),
                    start=word.get("start_time", 0) / 1000.0,
                    end=word.get("end_time", 0) / 1000.0,
                    speaker=word.get("speaker", "Speaker 0")
                ))
        
        usage = UsageInfo(
            type="duration",
            seconds=int(duration) if duration else None
        )
        
        response = TranscriptionResponse(
            text=text,
            task="transcribe",
            language=language,
            duration=duration,
            segments=formatted_segments if "segment" in timestamp_granularities or response_format == "verbose_json" else None,
            words=formatted_words if "word" in timestamp_granularities else None,
            usage=usage
        )
        
        if response_format == "json":
            return {
                "text": response.text,
                "usage": response.usage.model_dump() if response.usage else None
            }
        elif response_format == "verbose_json":
            # Return full response model for verbose JSON
            return response
        
        return response
    
    @staticmethod
    def _format_speaker_aware_text(sentences: List[Dict[str, Any]]) -> str:
        """Format text with speaker labels using sentence speaker mapping."""
        if not sentences:
            return ""
        
        result_parts = []
        previous_speaker = None
        
        for sentence_dict in sentences:
            speaker = sentence_dict.get("speaker", "Speaker 0")
            sentence_text = sentence_dict.get("text", "").strip()
            
            if not sentence_text:
                continue
                
            # If this speaker doesn't match the previous one, start a new speaker section
            if speaker != previous_speaker:
                if previous_speaker is not None:
                    result_parts.append("\n\n")
                result_parts.append(f"{speaker}: ")
                previous_speaker = speaker
            
            # Add the current sentence
            result_parts.append(sentence_text + " ")
        
        return "".join(result_parts).strip()
    
    @staticmethod
    def _format_srt(segments: List[Dict[str, Any]]) -> str:
        srt_content = []
        for i, segment in enumerate(segments, 1):
            start_time = ResponseFormatter._format_timestamp(
                segment.get("start_time", 0), 
                decimal_marker=","
            )
            end_time = ResponseFormatter._format_timestamp(
                segment.get("end_time", 0), 
                decimal_marker=","
            )
            speaker = segment.get("speaker", "Speaker 0")
            text = segment.get("text", "").strip()
            
            srt_content.append(f"{i}")
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(f"{speaker}: {text}")
            srt_content.append("")
        
        return "\n".join(srt_content)
    
    @staticmethod
    def _format_vtt(segments: List[Dict[str, Any]]) -> str:
        vtt_content = ["WEBVTT", ""]
        
        for segment in segments:
            start_time = ResponseFormatter._format_timestamp(
                segment.get("start_time", 0), 
                decimal_marker="."
            )
            end_time = ResponseFormatter._format_timestamp(
                segment.get("end_time", 0), 
                decimal_marker="."
            )
            speaker = segment.get("speaker", "Speaker 0")
            text = segment.get("text", "").strip()
            
            vtt_content.append(f"{start_time} --> {end_time}")
            vtt_content.append(f"{speaker}: {text}")
            vtt_content.append("")
        
        return "\n".join(vtt_content)
    
    @staticmethod
    def _format_timestamp(milliseconds: float, decimal_marker: str = ".") -> str:
        hours = int(milliseconds // 3_600_000)
        milliseconds -= hours * 3_600_000
        
        minutes = int(milliseconds // 60_000)
        milliseconds -= minutes * 60_000
        
        seconds = int(milliseconds // 1_000)
        milliseconds = int(milliseconds - seconds * 1_000)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"

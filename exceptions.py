class TranscriptionError(Exception):
    """Base exception for transcription errors"""
    pass


class AudioValidationError(TranscriptionError):
    """Raised when audio file validation fails"""
    pass


class ModelInitializationError(TranscriptionError):
    """Raised when model initialization fails"""
    pass


class ChunkProcessingError(TranscriptionError):
    """Raised when audio chunk processing fails"""
    pass


class DiarizationError(TranscriptionError):
    """Raised when speaker diarization fails"""
    pass


class UnsupportedFormatError(TranscriptionError):
    """Raised when audio format is not supported"""
    pass
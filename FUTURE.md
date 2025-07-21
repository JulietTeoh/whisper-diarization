# Whisper Diarization Server - Future Improvements & Roadmap

This document outlines potential improvements, bug fixes, and performance optimizations for the Whisper Diarization Server. The most critical performance issues have been resolved, and the server is now in a stable, production-ready state. The items below represent opportunities for further enhancement.

## 🚀 High-Impact Performance Optimizations

### 1. **Async Operations with Thread Pools**
- **Issue**: Core ML operations (transcription, diarization) are CPU/GPU-bound and currently run synchronously within async functions, which can block the FastAPI event loop under high load.
- **Optimization**: Offload these blocking tasks to a thread pool using `fastapi.concurrency.run_in_threadpool`.
- **Example**:
  ```python
  # In services/transcription.py
  from fastapi.concurrency import run_in_threadpool

  async def _process_single_audio(self, ...):
      full_transcript, whisper_segments, info = await run_in_threadpool(
          self.whisper_service.transcribe_audio, 
          audio_info["audio_file_path"], ...
      )
      # ... and so on for other blocking calls
  ```
- **Benefits**: Improves server responsiveness and allows it to handle more concurrent connections without becoming blocked.

### 2. **Request Batching for GPU Efficiency**
- **Issue**: The server currently processes each request independently. For models like Whisper, batching multiple requests together can significantly improve GPU throughput.
- **Optimization**: Implement a request batching and queuing system. This would involve collecting incoming requests for a short period (e.g., 100ms) and processing them as a single batch.
- **Benefits**: Can lead to a 40-60% improvement in GPU utilization and overall throughput under heavy load.

## 💡 Feature Enhancements

### 1. **Dynamic Speaker Diarization Toggle**
- **Current State**: Speaker diarization is enabled or disabled globally via the `.env` configuration.
- **Enhancement**: Add an `enable_speaker_diarization` boolean parameter to the `/v1/audio/transcriptions` endpoint. This would allow clients to decide whether to perform diarization on a per-request basis.
- **Implementation**:
  - Add the parameter to the `create_transcription` function in `server.py`.
  - Pass it down to the `TranscriptionService`.
  - Conditionally execute the `DiarizationService` based on this parameter, similar to how `enable_vocal_separation` is handled.

### 2. **Streaming Transcription with WebSockets**
- **Enhancement**: Implement a WebSocket endpoint for real-time streaming transcription. This would allow clients to send audio data in chunks and receive transcription results as they are generated.
- **Benefits**: Enables real-time applications, such as live captioning or meeting transcription.
- **Complexity**: High. Requires significant changes to the architecture to handle audio chunking, state management, and WebSocket communication.

### 3. **Intelligent Audio Chunking**
- **Current State**: Audio is chunked based on a fixed duration.
- **Enhancement**: Implement a more intelligent chunking strategy based on silence detection (e.g., using `librosa.effects.split`). This would create more natural breaks in the audio and could improve transcription accuracy at chunk boundaries.
- **Benefits**: Potentially better accuracy and more natural segmentation of the final transcript.

## 🔧 Code Quality & Maintenance

### 1. **Standardized Error Handling**
- **Issue**: While the server has basic error handling, it could be improved by using more specific custom exceptions and ensuring consistent HTTP status codes are returned.
- **Enhancement**: Define a clear hierarchy of custom exceptions (e.g., `ModelLoadError`, `AudioProcessingError`) and map them to appropriate HTTP responses in `server.py`.

### 2. **Configuration Validation**
- **Issue**: The configuration is loaded from `.env` but not rigorously validated on startup.
- **Enhancement**: Use Pydantic's validation capabilities more extensively in `config.py` to ensure that all required settings are present and have valid values when the server starts.

### 3. **API Compatibility**
- **Issue**: The API is compatible with the basic OpenAI transcription endpoint, but it could support more parameters from the official API.
- **Enhancement**: Add support for additional OpenAI API parameters like `timestamp_granularities` (word-level timestamps are already implemented, but could be exposed this way) to improve drop-in compatibility.

## 🧪 Testing

### 1. **Load Testing**
- **Recommendation**: Use tools like `locust` or `k6` to simulate concurrent users and measure the server's performance under load. This would help identify bottlenecks and determine the optimal number of concurrent workers.

### 2. **End-to-End Testing**
- **Recommendation**: Expand the test suite to include more comprehensive end-to-end tests that cover all API parameters and response formats. This could involve a set of sample audio files with known expected outputs.
import logging
import threading
from typing import Optional, Tuple
import torch

import faster_whisper
from ctc_forced_aligner import load_alignment_model
from deepmultilingualpunctuation import PunctuationModel
from pyannote.audio import Pipeline

from config import settings
from exceptions import ModelInitializationError

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Singleton class to manage all ML models used across the application.
    Ensures only one instance of each model type is loaded in memory.
    """
    
    _instance = None
    _lock = threading.Lock()
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not ModelManager._initialized:
            with ModelManager._lock:
                if not ModelManager._initialized:
                    self._initialize_models()
                    ModelManager._initialized = True
    
    def _initialize_models(self):
        """Initialize all models on first access"""
        logger.info("Initializing ModelManager - models will be loaded on demand...")

        # Model storage
        self.whisper_model = None
        self.whisper_pipeline = None
        self.alignment_model = None
        self.alignment_tokenizer = None
        self.punctuation_model = None
        self.pyannote_pipeline = None

        # Device configuration
        self.device = settings.device
        self.diarization_device = settings.diarization_device
        self.compute_type = settings.compute_type

        # Initialize models based on workflow configuration
        self._initialize_whisper_model()

        if settings.enable_speaker_diarization:
            self._initialize_alignment_model()
            self._initialize_pyannote_pipeline()

        if settings.enable_punctuation_restoration:
            self._initialize_punctuation_model()

        logger.info("ModelManager initialization completed successfully")

    def _initialize_whisper_model(self):
        """Initialize Whisper model and pipeline"""
        try:
            logger.info(f"Loading Whisper model: {settings.whisper_model} on {self.device}")
            self.whisper_model = faster_whisper.WhisperModel(
                settings.whisper_model,
                device=self.device,
                compute_type=self.compute_type
            )
            self.whisper_pipeline = faster_whisper.BatchedInferencePipeline(self.whisper_model)
            logger.info(f"Successfully loaded Whisper model: {settings.whisper_model} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize Whisper model: {e}")
            raise ModelInitializationError(f"Whisper model initialization failed: {e}")
    
    def _initialize_alignment_model(self):
        """Initialize alignment model for forced alignment"""
        try:
            logger.info(f"Loading alignment model on {self.diarization_device}")
            dtype = torch.float16 if self.diarization_device == "cuda" else torch.float32
            self.alignment_model, self.alignment_tokenizer = load_alignment_model(
                self.diarization_device, dtype=dtype,
            )
            logger.info(f"Successfully loaded alignment model on {self.diarization_device}")
        except Exception as e:
            logger.error(f"Failed to initialize alignment model: {e}")
            raise ModelInitializationError(f"Alignment model initialization failed: {e}")
    
    def _initialize_punctuation_model(self):
        """Initialize punctuation restoration model"""
        try:
            logger.info("Loading punctuation model...")
            self.punctuation_model = PunctuationModel(model="kredor/punctuate-all")
            logger.info("Successfully loaded punctuation model")
        except Exception as e:
            logger.error(f"Failed to initialize punctuation model: {e}")
            raise ModelInitializationError(f"Punctuation model initialization failed: {e}")

    def _initialize_pyannote_pipeline(self):
        """Initializes the Pyannote diarization pipeline."""
        try:
            logger.info(f"Loading Pyannote diarization pipeline on {self.diarization_device}")
            
            # Get HuggingFace token from settings
            hf_token = settings.huggingface_token or settings.hf_token
            if not hf_token:
                logger.warning("No HuggingFace token found. Pipeline will attempt to use cached models or public access.")
            
            # Initialize Pyannote pipeline
            model_name = "pyannote/speaker-diarization-3.1"
            self.pyannote_pipeline = Pipeline.from_pretrained(
                model_name,
                use_auth_token=hf_token
            )
            
            # Move to specified device
            self.pyannote_pipeline.to(torch.device(self.diarization_device))
            
            logger.info(f"Successfully loaded Pyannote diarization pipeline: {model_name} on {self.diarization_device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pyannote diarization pipeline: {e}")
            raise ModelInitializationError(f"Pyannote diarization pipeline initialization failed: {e}")
    
    # Whisper model access methods
    def get_whisper_model(self) -> faster_whisper.WhisperModel:
        """Get the shared Whisper model instance"""
        if self.whisper_model is None:
            self._initialize_whisper_model()
        if self.whisper_model is None:
            raise ModelInitializationError("Whisper model not initialized")
        return self.whisper_model

    def get_whisper_pipeline(self) -> faster_whisper.BatchedInferencePipeline:
        """Get the shared Whisper pipeline instance"""
        if self.whisper_pipeline is None:
            self._initialize_whisper_model()
        if self.whisper_pipeline is None:
            raise ModelInitializationError("Whisper pipeline not initialized")
        return self.whisper_pipeline

    # Alignment model access methods
    def get_alignment_model(self) -> Optional[Tuple[object, object]]:
        """
        Gets the shared alignment model and tokenizer, but only if speaker
        diarization is enabled in the settings.
        """
        if not settings.enable_speaker_diarization:
            return None, None
        if self.alignment_model is None or self.alignment_tokenizer is None:
            self._initialize_alignment_model()
        return self.alignment_model, self.alignment_tokenizer

    def get_punctuation_model(self) -> Optional[PunctuationModel]:
        """
        Gets the shared punctuation model, but only if punctuation restoration
        is enabled in the settings.
        """
        if not settings.enable_punctuation_restoration:
            return None
        if self.punctuation_model is None:
            self._initialize_punctuation_model()
        return self.punctuation_model

    # Pyannote diarization pipeline access methods
    def get_pyannote_pipeline(self) -> Pipeline:
        """Get the shared Pyannote diarization pipeline instance"""
        if self.pyannote_pipeline is None:
            self._initialize_pyannote_pipeline()
        if self.pyannote_pipeline is None:
            raise ModelInitializationError("Pyannote diarization pipeline not initialized")
        return self.pyannote_pipeline
    
    # Device information
    def get_whisper_device(self) -> str:
        """Get the device used for Whisper model"""
        return self.device
    
    def get_diarization_device(self) -> str:
        """Get the device used for diarization models"""
        return self.diarization_device
    
    def get_compute_type(self) -> str:
        """Get the compute type used for Whisper model"""
        return self.compute_type
    
    # Cleanup methods
    def cleanup(self):
        """Clean up all models and free memory"""
        logger.info("Cleaning up ModelManager...")
        
        try:
            if self.whisper_model is not None:
                del self.whisper_model
                self.whisper_model = None
            
            if self.whisper_pipeline is not None:
                del self.whisper_pipeline
                self.whisper_pipeline = None
            
            if self.alignment_model is not None:
                del self.alignment_model
                self.alignment_model = None
            
            if self.alignment_tokenizer is not None:
                del self.alignment_tokenizer
                self.alignment_tokenizer = None
            
            if self.punctuation_model is not None:
                del self.punctuation_model
                self.punctuation_model = None
            
            if self.pyannote_pipeline is not None:
                del self.pyannote_pipeline
                self.pyannote_pipeline = None
            
            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("ModelManager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during ModelManager cleanup: {e}")
    
    def get_memory_info(self) -> dict:
        """Get current memory usage information"""
        info = {
            "whisper_loaded": self.whisper_model is not None,
            "alignment_loaded": self.alignment_model is not None,
            "punctuation_loaded": self.punctuation_model is not None,
            "pyannote_pipeline_loaded": self.pyannote_pipeline is not None,
            "device": self.device,
            "diarization_device": self.diarization_device,
        }
        
        if torch.cuda.is_available():
            info["gpu_memory_allocated"] = torch.cuda.memory_allocated()
            info["gpu_memory_cached"] = torch.cuda.memory_reserved()
        
        return info


# Global singleton instance
model_manager = ModelManager()
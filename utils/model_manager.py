import logging
import threading
from typing import Optional, Tuple
import torch

import faster_whisper
from ctc_forced_aligner import load_alignment_model
from deepmultilingualpunctuation import PunctuationModel
from nemo.collections.asr.models.msdd_models import NeuralDiarizer

from config import settings
from exceptions import ModelInitializationError
from helpers import create_config

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
        self.nemo_diarizer = None
        self.nemo_base_temp_dir = None

        # Device configuration
        self.device = settings.device
        self.diarization_device = settings.diarization_device
        self.compute_type = settings.compute_type

        # Initialize models based on workflow configuration
        self._initialize_whisper_model()

        if settings.enable_speaker_diarization:
            self._initialize_alignment_model()
            self._initialize_nemo_diarizer()

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

    def _initialize_nemo_diarizer(self):
        """Initializes the NeMo diarization model."""
        try:
            logger.info(f"Loading NeMo diarization models on {self.diarization_device}")
            import tempfile
            temp_dir = settings.temp_dir if hasattr(settings, 'temp_dir') and settings.temp_dir else None
            self.nemo_base_temp_dir = tempfile.mkdtemp(dir=temp_dir)
            base_config = create_config(self.nemo_base_temp_dir)
            self.nemo_diarizer = NeuralDiarizer(cfg=base_config).to(self.diarization_device)
        except Exception as e:
            raise ModelInitializationError(f"NeMo diarization model initialization failed: {e}")
    
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

    # NeMo diarization model access methods
    def get_nemo_diarizer(self) -> NeuralDiarizer:
        """Get the shared NeMo diarization model instance"""
        if self.nemo_diarizer is None:
            self._initialize_nemo_diarizer()
        if self.nemo_diarizer is None:
            raise ModelInitializationError("NeMo diarization model not initialized")
        return self.nemo_diarizer
    
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
            
            if self.nemo_diarizer is not None:
                del self.nemo_diarizer
                self.nemo_diarizer = None
            
            # Clean up the base temp directory for NeMo
            if self.nemo_base_temp_dir is not None:
                import shutil
                import os
                if os.path.exists(self.nemo_base_temp_dir):
                    shutil.rmtree(self.nemo_base_temp_dir)
                    logger.info(f"Cleaned up NeMo base temp directory: {self.nemo_base_temp_dir}")
                self.nemo_base_temp_dir = None
            
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
            "nemo_diarizer_loaded": self.nemo_diarizer is not None,
            "device": self.device,
            "diarization_device": self.diarization_device,
        }
        
        if torch.cuda.is_available():
            info["gpu_memory_allocated"] = torch.cuda.memory_allocated()
            info["gpu_memory_cached"] = torch.cuda.memory_reserved()
        
        return info


# Global singleton instance
model_manager = ModelManager()
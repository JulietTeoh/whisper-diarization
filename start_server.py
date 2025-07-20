#!/usr/bin/env python3
"""
Startup script for the Whisper Diarization Server
"""

import os
import sys
import logging
from pathlib import Path

# Add the current directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from config import settings

def main():
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Whisper Diarization Server...")
    logger.info(f"Configuration:")
    logger.info(f"  - Host: {settings.host}")
    logger.info(f"  - Port: {settings.port}")
    logger.info(f"  - Whisper Model: {settings.whisper_model}")
    logger.info(f"  - Device: {settings.device}")
    logger.info(f"  - Chunking: {settings.chunk_length_minutes} minutes")
    logger.info(f"  - Stemming: {settings.enable_stemming}")
    
    # Check if HuggingFace token is set
    if settings.huggingface_token:
        logger.info("  - HuggingFace token: Set")
    else:
        logger.warning("  - HuggingFace token: Not set")
    
    try:
        import uvicorn
        from server import app
        
        uvicorn.run(
            app,
            host=settings.host,
            port=settings.port,
            log_level=settings.log_level.lower(),
            reload=settings.debug
        )
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
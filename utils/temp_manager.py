import os
import shutil
import tempfile
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class TempManager:
    def __init__(self, base_temp_dir: Optional[str] = None):
        self.base_temp_dir = base_temp_dir or tempfile.gettempdir()
        self.temp_dirs: Dict[str, str] = {}
        self.pid = os.getpid()
    
    def create_temp_dir(self, request_id: str) -> str:
        temp_dir = os.path.join(
            self.base_temp_dir, 
            f"whisper_diarization_{self.pid}_{request_id}"
        )
        os.makedirs(temp_dir, exist_ok=True)
        self.temp_dirs[request_id] = temp_dir
        logger.info(f"Created temp directory: {temp_dir}")
        return temp_dir
    
    def get_temp_dir(self, request_id: str) -> Optional[str]:
        return self.temp_dirs.get(request_id)
    
    def cleanup_temp_dir(self, request_id: str) -> None:
        temp_dir = self.temp_dirs.get(request_id)
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temp directory: {temp_dir}")
            except Exception as e:
                logger.error(f"Failed to cleanup temp directory {temp_dir}: {e}")
            finally:
                self.temp_dirs.pop(request_id, None)
    
    def cleanup_all(self) -> None:
        for request_id in list(self.temp_dirs.keys()):
            self.cleanup_temp_dir(request_id)
import psutil
from app.utils.logger import logger

class ResourceManager:
    """Monitor and manage system resources"""
    
    def __init__(self, memory_threshold_mb=1024):
        self.memory_threshold = memory_threshold_mb * 1024 * 1024  # Convert to bytes
    
    def check_resources(self) -> bool:
        """Check if system resources are within acceptable limits
        """
        try:
            memory_used = psutil.Process().memory_info().rss
            if memory_used > self.memory_threshold:
                logger.warning(
                    f"Memory usage ({memory_used/1024/1024:.2f}MB) "
                    f"exceeded threshold ({self.memory_threshold/1024/1024:.2f}MB)"
                )
                return False
            return True
        except Exception as e:
            logger.error(f"Error checking resources: {str(e)}")
            return False 
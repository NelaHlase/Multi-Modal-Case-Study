import os           #will handles the environment variables
import logging
import time
from functools import wraps
from dataclasses import dataclass
from typing import Dict, Optional
from contextlib import contextmanager

#logging is a module that allows you to log messages to the console or a file. It tracks down issues in production by providing an execution trail when errors occur
#this is useful in for debugging, performance monitoring, and error tracking which is very important for large scale applications.

# Get log level from environment variable, default to INFO if not set
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}
current_log_level = LOG_LEVELS.get(log_level, logging.INFO)

# Configure logger
logger = logging.getLogger('image_retrieval')
logger.setLevel(current_log_level)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(current_log_level)

# Create file handler
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(current_log_level)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

@dataclass
class PerformanceMetrics:
    """Class to store and calculate performance metrics"""
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    
    def complete(self):
        """Complete the timing and calculate duration"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        logger.info(f"{self.operation_name} completed in {self.duration:.2f} seconds")
        return self.duration

    @staticmethod
    def start_operation(name: str) -> 'PerformanceMetrics':
        """Start timing a new operation"""
        return PerformanceMetrics(name, time.time())

def performance_logger(operation_name):
    """Decorator to log performance metrics of operations
    
    Args:
        operation_name (str): Name of the operation being timed
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            duration = end_time - start_time
            logger.info(f"{operation_name} completed in {duration:.2f} seconds")
            
            return result
        return wrapper
    return decorator

@contextmanager
def monitor_performance(operation_name: str):
    """Context manager to monitor performance of operations
    
    Args:
        operation_name (str): Name of the operation being monitored
    
    Yields:
        PerformanceMetrics: Performance metrics object for the operation
    """
    metrics = PerformanceMetrics.start_operation(operation_name)
    try:
        yield metrics
    finally:
        metrics.complete()
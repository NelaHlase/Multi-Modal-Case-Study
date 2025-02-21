import pytest
import time
from app.utils.logger import (
    logger,
    performance_logger,
    PerformanceMetrics,
    monitor_performance
)

def test_performance_metrics_initialization():
    """Test the initialization of PerformanceMetrics class
    
    Verifies that a new PerformanceMetrics instance is created with correct initial values
    
    """

    metrics = PerformanceMetrics.start_operation("test_operation")
    assert metrics.operation_name == "test_operation"       
    assert metrics.start_time is not None                   
    assert metrics.end_time is None                         
    assert metrics.duration is None                         

def test_performance_logger_decorator():
    """Test the performance_logger decorator functionality
    
    Ensures that the decorator doesn't interfere with the function's return value
    
    """

    @performance_logger("test_operation")
    def test_func():
        return "test"
    
    result = test_func()
    assert result == "test"

def test_logger_levels(caplog):
    """Test different logging levels
    
    Verifies that the logger correctly handles different logging levels
    Uses pytest's caplog fixture to capture log output
    
    """

    test_message = "Test log message"
    
    logger.debug(test_message)
    logger.info(test_message)
    logger.warning(test_message)
    logger.error(test_message)
    
    assert any(record.levelname == 'INFO' and test_message in record.message 
              for record in caplog.records)
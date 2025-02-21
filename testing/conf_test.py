import pytest
import os
import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

@pytest.fixture
def sample_images():
    """Fixture providing paths to test images
    """
    test_images_dir = Path(__file__).parent / "test_data" / "images"
    return list(test_images_dir.glob("*.jpg"))

@pytest.fixture
def mock_performance_metrics():
    """Fixture providing a clean PerformanceMetrics instance
    """
    from app.utils.logger import PerformanceMetrics
    return PerformanceMetrics()
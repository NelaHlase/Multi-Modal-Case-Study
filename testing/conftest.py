import pytest
import os
from PIL import Image
import tempfile

@pytest.fixture
def sample_images():
    """Fixture to create temporary test images
    """
    temp_dir = tempfile.mkdtemp()
    image_paths = []
    
    for i in range(3):
        img = Image.new('RGB', (100, 100), color=f'rgb({i*50}, {i*50}, {i*50})')
        path = os.path.join(temp_dir, f'test_image_{i}.jpg')
        img.save(path)
        image_paths.append(path)
    
    yield image_paths  
    
    # Cleans up the temporary images
    for path in image_paths:
        os.remove(path)
    os.rmdir(temp_dir) 
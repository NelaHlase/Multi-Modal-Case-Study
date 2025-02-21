import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from PIL import Image
from tqdm import tqdm
from app.utils.logger import logger


class ImageDataset:
    """
    Handles loading and managing the dataset
    """
    def __init__(self, file_path, num_images):
        """
        Initializes the ImageDataset with the path and number of images to load.
        
        Args:
            file_path (str): Path to the directory containing images.
            num_images (int): Number of images to load from the directory.
        """
        print(f"Attempting to initialize with path: {file_path}")
        self.file_path = os.path.abspath(os.path.normpath(file_path))
        self.num_images = num_images
        self.images = []
        self.image_features = None

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Data path not found: {self.file_path}")

    def load_images(self):
        """
        Loads images from the specified directory.
        
        Raises:
            ValueError: If no image files are found in the specified directory.
        """
        logger.info(f"Loading from directory: {self.file_path}")
        
        valid_extensions = ('.jpg', '.jpeg', '.png')
        
        image_files = [f for f in os.listdir(self.file_path)
                      if f.lower().endswith(valid_extensions)][:self.num_images]
        
        if not image_files:
            raise ValueError(f"No image files found in {self.file_path}")
        
        logger.info(f"Loading {len(image_files)} images...")
        
        for img_file in tqdm(image_files):
            try:
                img_path = os.path.join(self.file_path, img_file)
                image = Image.open(img_path).convert('RGB')
                self.images.append({
                    'path': img_path,
                    'image': image,
                    'filename': img_file
                })
            except Exception as e:
                logger.error(f"Error loading image {img_file}: {str(e)}")
        
        logger.info(f"Successfully loaded {len(self.images)} images")

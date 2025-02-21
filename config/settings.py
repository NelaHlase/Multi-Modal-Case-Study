import os
import kagglehub
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """Application settings loaded from environment variables
    """
    
    MODEL_NAME = os.getenv('MODEL_NAME', 'openai/clip-vit-base-patch32')
    DATASET_PATH = kagglehub.dataset_download("alessandrasala79/ai-vs-human-generated-dataset")
    FILE_PATH = os.path.join(DATASET_PATH, "test_data_v2")  # Path to the images directory
    NUM_IMAGES = int(os.getenv('NUM_IMAGES', 500))
    TOP_K = int(os.getenv('TOP_K', 5))
    MAX_BATCH_SIZE = int(os.getenv('MAX_BATCH_SIZE', 32))
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
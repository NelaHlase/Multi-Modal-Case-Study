import os
import sys
import logging
import torch
import gradio as gr
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from app.utils.logger import logger
from config.settings import Settings
from app.core.retriever import ImageRetriever
from app.core.image_dataset import ImageDataset
from app.interface.web_interface import create_interface
from app.core.model_loader import load_model, preprocess_images

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dataset = None    
retriever = None

def search_images(query):
    """Search function for the interface"""
    try:
        if not query:
            return []
        logger.info(f"Searching for: {query}")
        
        # Add timeout and limit processing
        if not dataset or not dataset.image_features:
            logger.error("Dataset or image features not initialized")
            return []
            
        text_features = retriever.encode_text(query)
        if text_features is None:
            logger.warning("Failed to encode text query")
            return []
            
        indices, scores = retriever.find_matches(
            text_features,
            dataset.image_features,
            Settings.TOP_K
        )
        
        # Validate results before returning
        valid_paths = []
        for idx in indices:
            if 0 <= idx < len(dataset.images):
                path = dataset.images[idx]['path']
                if os.path.exists(path):
                    valid_paths.append(path)
                    
        logger.info(f"Found {len(valid_paths)} valid matches")
        return valid_paths
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return []

def main():
    """Main function to initialize and run the system"""
    global dataset, retriever
    try:
        logger.info("Initializing system...")
        
        # Initialize dataset
        logger.info("Loading dataset...")
        dataset = ImageDataset(Settings.FILE_PATH, Settings.NUM_IMAGES)
        dataset.load_images()
        
        # Load model and processor
        logger.info("Loading CLIP model...")
        model = CLIPModel.from_pretrained(Settings.MODEL_NAME)
        processor = CLIPProcessor.from_pretrained(Settings.MODEL_NAME)
        
        # Process images and store features
        logger.info("Processing images...")
        with torch.no_grad():
            processed_images = processor(
                images=[img['image'] for img in dataset.images],
                return_tensors="pt",
                padding=True
            )
            dataset.image_features = model.get_image_features(**processed_images)
        
        # Initialize retriever
        logger.info("Initializing retriever...")
        retriever = ImageRetriever(model, processor)

        # Create and launch interface
        logger.info("Creating interface...")
        interface = create_interface(dataset, retriever)
        interface.launch(
            inbrowser=True,
            allowed_paths=[os.path.dirname(Settings.FILE_PATH)]  # Add dataset directory to allowed paths
        )
    
    except Exception as e:
        logger.error(f"System initialization failed: {str(e)}")

if __name__ == "__main__":
    main()



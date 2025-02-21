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
        text_features = retriever.encode_text(query)
        indices, scores = retriever.find_matches(
            text_features,
            dataset.image_features,
            Settings.TOP_K
        )
        for idx, (index, score) in enumerate(zip(indices, scores), 1):
            logger.info(f"Match {idx}: Score = {score:.4f}, Image = {dataset.images[index]['path']}")
        return [dataset.images[idx]['path'] for idx in indices]
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
        interface.launch(inbrowser=True)
    
    except Exception as e:
        logger.error(f"System initialization failed: {str(e)}")

if __name__ == "__main__":
    main()



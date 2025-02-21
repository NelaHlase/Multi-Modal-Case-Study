from transformers import CLIPProcessor, CLIPModel
from config.settings import Settings
from app.utils.logger import logger


def load_model():
    """Load CLIP model and processor
    """
    model = CLIPModel.from_pretrained(Settings.MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(Settings.MODEL_NAME)
    return model, processor


def preprocess_images(dataset, model, processor):
    """
    Generates and stores image embeddings for all loaded images
    """
    import torch
    logger.info("Generating image embeddings...")
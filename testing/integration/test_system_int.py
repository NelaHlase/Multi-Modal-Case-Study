import pytest
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from app.core.retriever import ImageRetriever
from app.utils.logger import performance_logger
from app.interface.web_interface import create_interface
from config.settings import Settings
import os

@pytest.fixture
def model_and_processor():
    """Fixture to provide model and processor
    """
    model = CLIPModel.from_pretrained(Settings.MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(Settings.MODEL_NAME)
    return model, processor

def test_end_to_end_image_retrieval(sample_images, model_and_processor):
    """Test the complete image retrieval pipeline (loading, processing and search functionality)
    """
    model, processor = model_and_processor
    retriever = ImageRetriever(model=model, processor=processor)
    
    # Load and process sample images
    images = [Image.open(img_path) for img_path in sample_images]
    processed_images = processor(images=images, return_tensors="pt")
    image_features = model.get_image_features(**processed_images)
    
    @performance_logger('test_retrieval')
    def perform_search(query):
        if not query.strip():  # Check for empty or whitespace-only query
            return []
        text_features = retriever.encode_text(query)
        return retriever.find_matches(text_features, image_features, top_k=2)
    
    # Test with valid query
    results = perform_search("a girl climbing a tree")
    assert len(results) > 0
    
    # Test with edge case query
    results = perform_search("")
    assert len(results) == 0

def test_gradio_interface_integration(sample_images, model_and_processor):
    """Test the Gradio interface integration"""
    
    model, processor = model_and_processor
    retriever = ImageRetriever(model=model, processor=processor)
    
    # Create dataset with sample images directory
    from app.core.image_dataset import ImageDataset
    
    # Get the directory containing the sample images
    image_dir = os.path.dirname(sample_images[0])
    dataset = ImageDataset(image_dir, len(sample_images))
    dataset.load_images()
    
    # Create interface
    interface = create_interface(dataset, retriever)
    
    # Test the search function used by the interface
    from main import search_images
    result = search_images("test query")
    assert isinstance(result, list)
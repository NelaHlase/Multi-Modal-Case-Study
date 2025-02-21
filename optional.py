import os
import sys
import torch
import gradio as gr
from transformers import CLIPModel, CLIPProcessor, pipeline
import logging
from typing import List, Optional
import speech_recognition as sr
from gtts import gTTS
import io
import tempfile

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from app.core.image_dataset import ImageDataset
from app.core.retriever import ImageRetriever
from app.utils.logger import logger
from config.settings import Settings

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
        
        results = retriever.search(query)
        return results
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return []

def create_interface(dataset, retriever):
    """Create the Gradio interface with CSS custom styling"""
    
    # Define CSS styling
    custom_css = """
        body {
            background-color: #191970;
        }
        .main-container {
            margin: 0 auto;
            max-width: 900px;
            padding: 20px;
        }
        .banner-container {
            text-align: center;
            margin-bottom: 20px;
        }
        .title {
            font-size: 24px;
            font-weight: bold;
            margin: 20px 0;
            text-align: center;
            color: white;
        }
        .search-container {
            background-color: peru;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    """

    def search_images(query, use_voice=False):
        """Search function with optional voice output"""
        try:
            if not query:
                return [], None
            
            logger.info(f"Searching for: {query}")
            results = retriever.search(query)
            
            if use_voice and results:
                # Generate descriptions for voice output
                descriptions = [f"Image {i+1}: {os.path.basename(path)}" 
                              for i, path in enumerate(results)]
                text = "Found the following images. " + " ".join(descriptions)
                audio_path = text_to_speech(text)
                return results, audio_path
            
            return results, None
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return [], None

    def text_to_speech(text: str) -> Optional[str]:
        """Convert text to speech and save to temporary file"""
        try:
            # Create a temporary file
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"speech_{hash(text)}.mp3")
            
            # Generate and save the speech
            tts = gTTS(text=text, lang='en')
            tts.save(temp_path)
            
            return temp_path
            
        except Exception as e:
            logger.error(f"Text-to-speech error: {str(e)}")
            return None

    # Create interface with custom styling
    interface = gr.Blocks(css=custom_css)
    
    with interface:
        with gr.Column(elem_classes="main-container"):
            
            # Title
            gr.Markdown(
                "# AI vs Human Generated Images",
                elem_classes="title"
            )
            
            # Search container
            with gr.Column(elem_classes="search-container"):
                query_input = gr.Textbox(
                    label="Enter an image description",
                    placeholder="Type your search query here...",
                    elem_id="search-input"
                )
                use_voice = gr.Checkbox(
                    label="Enable voice output",
                    value=False
                )
                search_button = gr.Button(
                    "Search",
                    variant="primary",
                    elem_id="search-button"
                )
                gallery = gr.Gallery(
                    columns=2,
                    height="auto",
                    elem_id="results-gallery"
                )
                audio_output = gr.Audio(
                    label="Voice Description",
                    type="filepath",
                    visible=True
                )
                
                # Event handlers
                search_button.click(
                    fn=search_images,
                    inputs=[query_input, use_voice],
                    outputs=[gallery, audio_output]
                )
                query_input.submit(
                    fn=search_images,
                    inputs=[query_input, use_voice],
                    outputs=[gallery, audio_output]
                )
    
    return interface

def validate_file_path(file_path):
    """Validate that the file path is within allowed directories"""
    allowed_dir = os.path.abspath(Settings.FILE_PATH)
    target_path = os.path.abspath(file_path)
    return os.path.commonpath([allowed_dir]) == os.path.commonpath([allowed_dir, target_path])

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
            image_features = model.get_image_features(**processed_images)
            dataset.image_features = image_features  # Store in dataset
        
        # Initialize retriever
        logger.info("Initializing retriever...")
        retriever = ImageRetriever(model, processor)
        # Explicitly set these attributes
        retriever.image_features = image_features  # Use the local variable 
        retriever.image_paths = [img['path'] for img in dataset.images]
        
        # Create and launch interface
        logger.info("Creating interface...")
        interface = create_interface(dataset, retriever)
        
        logger.info("Launching interface...")
        interface.launch(
            server_name="127.0.0.1",
            server_port=None,  # Auto-select available port
            share=False,
            inbrowser=True
        )
    
    except Exception as e:
        logger.error(f"System initialization failed: {str(e)}")
        raise 

if __name__ == "__main__":
    main()
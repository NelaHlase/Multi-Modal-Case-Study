import os
import gradio as gr
from app.utils.logger import logger
from config.settings import Settings


def create_interface(dataset, retriever):
    """Create the Gradio interface with custom styling"""
    
    # Define the CSS for background color and styling
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
    
    def search_images(query):
        """Search function for the interface"""
        if not query:
            return []
        text_features = retriever.encode_text(query)
        indices, _ = retriever.find_matches(
            text_features,
            dataset.image_features,
            Settings.TOP_K
        )
        return [dataset.images[idx]['path'] for idx in indices]

    # Create the interface with banner and styling
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
                
                # Event handlers
                search_button.click(
                    fn=search_images,
                    inputs=query_input,
                    outputs=gallery
                )
                query_input.submit(
                    fn=search_images,
                    inputs=query_input,
                    outputs=gallery
                )
    
    return interface
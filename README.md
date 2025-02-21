# Multimodal Image Retrieval System

## Project Overview

This projects implements a multimodal image retrieval system using the CLIP model that takes a user (in text describing an image) query and retrieves the most relevant images from a dataset pertaining to the query. It is designed to be used in a Gradio web interface and can optionally provide a voice output of the results. CLIP and Gradio are both open-source models and libraries.

## Features

- Model combines natural language and image processing
- Supports multiple image formats (JPG, JPEG, PNG)
- Configurable number of results generated (top-K matches)
- Progress tracking for image loading and processing
- Efficient similarity search using normalized embeddings
- Voice output capabilities for accessibility (optional)

## Technologies Used

- CLIP (openai/clip-vit-base-patch32)
- Gradio for web interface
- PyTorch for tensor computations
- Transformers library
- Speech Recognition (speech_recognition)
- gTTS (Google Text-to-Speech)
- Additional libraries: PIL, tqdm, logging, os, sys, tempfile, psutil, dotenv

## Prerequisites

- Python 3.8 or higher
- Pip package manager
- At least 4GB of RAM
- CUDA-capable GPU (optional, but recommended for better performance)

## Setup and Installation

1. Clone the repository
2. Create a virtual environment 
    - `python -m venv venv`
    - Windows: `.\venv\Scripts\activate` or `venv\Scripts\activate.bat`
    - Linux/MacOS: `source venv/bin/activate`
3. Install the dependencies
    - `pip install -r requirements.txt` or `pip install torch torchvision transformers gradio Pillow tqdm python-dotenv kaggle gTTs SpeechRecognition pytest`
    - `pip install --upgrade pip`

## Running System

- Run the application using `python main.py` from the root directory
- Run the optional application using `python optional.py` from the root directory
- The application will be available at `http://127.0.0.1:7860`

## Testing

1. Run the tests using `pytest` from the testing directory
    - `pytest.ini` is the configuration file for the tests
    - `conftest.py` contains fixtures for the tests
    - `test_system_int.py` contains integration tests 
    - `test_logger.py` contains unit tests
2. Ensure that the tests are passing

## System Architecture 

The system architecture consists of three layers: Frontend, Application, and Infrastructure. 

### Frontend

The user interacts with the system through a web interface, built with Gradio, by inputting natural language queries. The interface presents the retrieved top-K images in a gallery format and manages the user experience.

### Application

The application layer is responsible for processing the user's query and retrieving the most relevant images from the dataset. The user's query is validated and sanitized using a query validator. It uses the CLIP model to generate embeddings for the query and the images, and then uses a similarity search to retrieve the most relevant images to the query. Images are processed in batches then converted to RGB format. 

### Infrastructure

The infrastructure layer is responsible for managing the system's resources and dependencies. It istores both images and their pre-computed CLIP embeddings. It maintains system logs and handles loading the model. 

## Assumptions

- The user will provide a valid query
- All required dependencies are installed
- The system will have access to a CUDA-capable GPU for faster processing

## Troubleshooting

- FileNotFoundError: Ensure your image directory path is correct in settings.py or the .env file
- CUDA out of memory: Reduce NUM_IMAGES in .env or use CPU mode
- Import errors: Verify all dependencies are installed via pip install -r requirements.txt

## Contributing

1. Fork the repository
2. Create a new branch
3. Make your changes and commit them
4. Push your changes to your fork
5. Create a pull request

## License

Not Applicable 

## Contact

For questions or issues, please contact [nelahlase@gmail.com].










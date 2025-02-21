
import os
import sys
import traceback
import torch
from PIL import Image
from functools import wraps
from transformers import CLIPModel, CLIPProcessor
from app.utils.logger import logger
from main import retriever, dataset
from config.settings import Settings

class ApplicationError(Exception):
    """Base exception class for application-specific errors"""
    def __init__(self, message, error_code=None, original_exception=None):
        self.message = message
        self.error_code = error_code
        self.original_exception = original_exception
        super().__init__(self.message)

class ModelError(ApplicationError):
    """Errors related to model operations"""
    pass

class DatasetError(ApplicationError):
    """Errors related to dataset operations"""
    pass

class ResourceError(ApplicationError):
    """Errors related to system resources"""
    pass

class ValidationError(ApplicationError):
    """Errors related to input validation"""
    pass

def handle_exceptions(func):
    """Decorator to handle exceptions in a standard way"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ApplicationError as e:
            logger.error(f"{type(e).__name__}: {e.message}")
            if e.original_exception:
                logger.debug(f"Original exception: {e.original_exception}")
            return None
        except torch.cuda.OutOfMemoryError as e:
            logger.critical(f"GPU out of memory: {str(e)}")
            # Fall back to CPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU memory cleared, falling back to CPU")
            return None
        except FileNotFoundError as e:
            logger.error(f"File not found: {str(e)}")
            return None
        except PermissionError as e:
            logger.error(f"Permission denied: {str(e)}")
            return None
        except (torch.SerializationError, RuntimeError) as e:
            logger.error(f"Model error: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            logger.debug(traceback.format_exc())
            return None
    return wrapper

def safe_file_access(file_path, operation_func, *args, **kwargs):
    """Safely access files with proper error handling
    
    Args:
        file_path (str): Path to the file
        operation_func (callable): Function to perform on the file
        
    Returns:
        The result of operation_func or None if error
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Path does not exist: {file_path}")
            
        return operation_func(file_path, *args, **kwargs)
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        return None
    except PermissionError as e:
        logger.error(f"Permission denied for file {file_path}: {str(e)}")
        return None
    except IsADirectoryError:
        logger.error(f"Expected file but found directory: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error accessing file {file_path}: {str(e)}")
        logger.debug(traceback.format_exc())
        return None

# 2. Update main.py with error handling
def search_images_with_error_handling(query):
    """Search function with robust error handling"""
    global retriever, dataset  # Add this line
    try:
        if not query:
            return []
            
        # Validate query
        from app.utils.query_validator import validate_query
        try:
            sanitized_query = validate_query(query)
        except ValueError as e:
            logger.warning(f"Query validation failed: {str(e)}")
            return []

        # Check system resources
        from app.core.resource_manager import ResourceManager
        resource_mgr = ResourceManager()
        if not resource_mgr.check_resources():
            logger.warning("Insufficient resources for image search")
            return []
            
        # Encode text with timeout protection
        try:
            text_features = retriever.encode_text(sanitized_query)
            if text_features is None:
                logger.warning("Text encoding failed")
                return []
        except torch.cuda.OutOfMemoryError:
            logger.warning("GPU memory exceeded during encoding, falling back to CPU")
            # Try to move model to CPU and retry
            retriever.model.to("cpu")
            text_features = retriever.encode_text(sanitized_query)
            
        # Find matches with timeout and error handling
        try:
            indices, scores = retriever.find_matches(
                text_features,
                dataset.image_features,
                Settings.TOP_K
            )
        except Exception as e:
            logger.error(f"Error finding matches: {str(e)}")
            return []
            
        # Verify indices are valid
        valid_indices = [idx for idx in indices if 0 <= idx < len(dataset.images)]
        if len(valid_indices) < len(indices):
            logger.warning(f"Found {len(indices) - len(valid_indices)} invalid indices")
            
        # Get image paths safely
        results = []
        for idx in valid_indices:
            try:
                path = dataset.images[idx]['path']
                if os.path.exists(path):
                    results.append(path)
                else:
                    logger.warning(f"Image path no longer exists: {path}")
            except (IndexError, KeyError) as e:
                logger.error(f"Error accessing image data: {str(e)}")
                
        return results
        
    except Exception as e:
        logger.error(f"Unhandled error in search: {str(e)}")
        logger.debug(traceback.format_exc())
        return []

# 3. Update app/core/model_loader.py with robust loading
def load_model_robust():
    """Load CLIP model with robust error handling and fallbacks"""
    try:
        # Try loading model with cuda
        if torch.cuda.is_available():
            try:
                logger.info(f"Loading model {Settings.MODEL_NAME} to GPU")
                model = CLIPModel.from_pretrained(Settings.MODEL_NAME)
                model.to("cuda")
                processor = CLIPProcessor.from_pretrained(Settings.MODEL_NAME)
                return model, processor
            except torch.cuda.OutOfMemoryError as e:
                logger.warning(f"GPU memory exceeded during model loading: {str(e)}")
                # Clear GPU memory
                torch.cuda.empty_cache()
                
        # Fall back to CPU
        logger.info(f"Loading model {Settings.MODEL_NAME} to CPU")
        model = CLIPModel.from_pretrained(Settings.MODEL_NAME)
        processor = CLIPProcessor.from_pretrained(Settings.MODEL_NAME)
        return model, processor
        
    except (OSError, IOError) as e:
        logger.error(f"Network error loading model: {str(e)}")
        raise ModelError("Failed to download model", original_exception=e)
    except ValueError as e:
        logger.error(f"Invalid model configuration: {str(e)}")
        raise ModelError("Invalid model configuration", original_exception=e)
    except Exception as e:
        logger.error(f"Unexpected error loading model: {str(e)}")
        logger.debug(traceback.format_exc())
        raise ModelError("Model loading failed", original_exception=e)

# 4. Update app/core/image_dataset.py with robust image loading
def load_image_robust(img_path):
    """Load a single image with robust error handling"""
    try:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        image = Image.open(img_path).convert('RGB')
        return image
    except FileNotFoundError as e:
        logger.warning(f"Image file not found: {str(e)}")
        return None
    except PermissionError as e:
        logger.warning(f"Permission denied for image: {str(e)}")
        return None
    except Image.DecompressionBombError as e:
        logger.warning(f"Image too large: {str(e)}")
        return None
    except Image.UnidentifiedImageError as e:
        logger.warning(f"Unidentified image format: {str(e)}")
        return None
    except Exception as e:
        logger.warning(f"Error loading image {img_path}: {str(e)}")
        return None

# 5. Add retry functionality for network operations
def retry_operation(func, max_attempts=3, delay=1):
    """Retry an operation with exponential backoff
    
    Args:
        func (callable): Function to retry
        max_attempts (int): Maximum number of retry attempts
        delay (int): Initial delay between retries in seconds
        
    Returns:
        The result of the function or raises the last exception
    """
    import time
    
    attempts = 0
    last_exception = None
    
    while attempts < max_attempts:
        try:
            return func()
        except (ConnectionError, TimeoutError, OSError) as e:
            attempts += 1
            last_exception = e
            wait_time = delay * (2 ** (attempts - 1))  # Exponential backoff
            logger.warning(f"Operation failed, retrying in {wait_time}s ({attempts}/{max_attempts}): {str(e)}")
            time.sleep(wait_time)
    
    logger.error(f"Operation failed after {max_attempts} attempts: {str(last_exception)}")
    raise last_exception
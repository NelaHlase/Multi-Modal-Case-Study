import torch
from transformers import CLIPProcessor, CLIPModel
from config.settings import Settings 
from typing import List

class ImageRetriever:
    def __init__(self, model, processor, image_features=None, image_paths=None):
        self.model = model
        self.processor = processor
        self.image_features = image_features
        self.image_paths = image_paths

    def search(self, query):
        """Search for images matching the query
        """
        if self.image_features is None:
            raise ValueError("Image features not available. Ensure dataset is loaded and preprocessed.")
            
        text_features = self.encode_text(query)
        indices, _ = self.find_matches(text_features, self.image_features, Settings.TOP_K)
        return [self.image_paths[idx] for idx in indices]

    def encode_text(self, text):
        """Encode text query to features
        """
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        return self.model.get_text_features(**inputs)

    def find_matches(self, text_features, image_features, top_k):
        """Find top-k matching images
        """
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        similarity = torch.mm(text_features, image_features.T)
        
        values, indices = similarity[0].topk(top_k)
        return indices.tolist(), values.tolist()
import os
import faiss
import json
import numpy as np
from PIL import Image
from src.clip_model import CLIPEmbedder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INDEX_PATH = os.path.join(BASE_DIR, "data", "embeddings.faiss")
META_PATH = os.path.join(BASE_DIR, "data", "metadata.json")

class HorseRecognizer:
    def __init__(self):
        self.embedder = CLIPEmbedder()
        self.index = faiss.read_index(INDEX_PATH)

        with open(META_PATH) as f:
            self.metadata = json.load(f)

    def recognize(self, image: Image.Image):
        vec = self.embedder.embed_image(image).astype("float32")
        scores, indices = self.index.search(vec, k=1)

        horse = self.metadata[indices[0][0]]
        confidence = float(scores[0][0])

        return horse, confidence

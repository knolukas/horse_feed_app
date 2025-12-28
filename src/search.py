import os
import faiss
import json
import numpy as np
from PIL import Image
from src.clip_model import CLIPEmbedder
from src.dinov2_model import DINOv2Embedder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INDEX_PATH = os.path.join(BASE_DIR, "data", "index.faiss")
META_PATH = os.path.join(BASE_DIR, "data", "metadata.json")

class HorseRecognizer:
    def __init__(self):
        self.embedder = DINOv2Embedder()
        self.index = faiss.read_index(INDEX_PATH)
        with open(META_PATH) as f:
            self.metadata = json.load(f)

    def recognize(self, image, top_k=3):
        vec = self.embedder.embed_image(image).astype("float32")
        faiss.normalize_L2(vec)

        # Sicherheit: nicht mehr Nachbarn als Pferde
        k = min(top_k, len(self.metadata))

        scores, indices = self.index.search(vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):

            # FAISS kann -1 zurückgeben
            if idx < 0 or idx >= len(self.metadata):
                continue

            results.append({
                "horse": self.metadata[idx],
                "confidence": float(score)
            })

        return results



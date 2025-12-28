import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


class DINOv2Embedder:
    def __init__(self, model_name="facebook/dinov2-base"):
        # Gerät (CPU/GPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Processor & Model laden
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def embed_image(self, image: Image.Image) -> np.ndarray:
        # Bild preprocessen
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        # Forward Pass
        with torch.no_grad():
            out = self.model(**inputs)

        # CLS Token auslesen (globales Embedding)
        emb = out.last_hidden_state[:, 0]

        # Normieren (wichtig für FAISS)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)

        return emb.cpu().numpy()
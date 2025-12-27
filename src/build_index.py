import os
import json
import faiss
import numpy as np
from PIL import Image
from src.clip_model import CLIPEmbedder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGE_DIR = os.path.join(BASE_DIR, "data", "images")
INDEX_PATH = os.path.join(BASE_DIR, "data", "embeddings.faiss")
META_PATH = os.path.join(BASE_DIR, "data", "metadata.json")

embedder = CLIPEmbedder()
vectors = []
metadata = []

for horse_name in os.listdir(IMAGE_DIR):
    horse_path = os.path.join(IMAGE_DIR, horse_name)
    if not os.path.isdir(horse_path):
        continue

    for img_name in os.listdir(horse_path):
        img_path = os.path.join(horse_path, img_name)
        image = Image.open(img_path).convert("RGB")
        vec = embedder.embed_image(image)

        vectors.append(vec[0])
        metadata.append(horse_name)

vectors = np.array(vectors).astype("float32")
index = faiss.IndexFlatIP(vectors.shape[1])
index.add(vectors)

faiss.write_index(index, INDEX_PATH)

with open(META_PATH, "w") as f:
    json.dump(metadata, f)

print("✅ FAISS Index gebaut!")

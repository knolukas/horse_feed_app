import os
import json
import faiss
import numpy as np
from PIL import Image
#from clip_model import CLIPEmbedder
from src.dinov2_model import DINOv2Embedder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(BASE_DIR, "data", "images")
INDEX_PATH = os.path.join(BASE_DIR, "data", "index.faiss")
META_PATH = os.path.join(BASE_DIR, "data", "metadata.json")

def build_index():

    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
        print(f"⚠️ {IMAGE_DIR} existierte noch nicht – wurde angelegt")
        return

    embedder = DINOv2Embedder()
    #embedder = CLIPEmbedder()
    horse_vectors = []
    horse_names = []

    for horse_name in os.listdir(IMAGE_DIR):
        horse_dir = os.path.join(IMAGE_DIR, horse_name)
        if not os.path.isdir(horse_dir):
            continue

        embeddings = []

        for img_file in os.listdir(horse_dir):
            if img_file.lower().endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(horse_dir, img_file)
                img = Image.open(img_path).convert("RGB")
                emb = embedder.embed_image(img)
                embeddings.append(emb)

        if not embeddings:
            continue

        # 🔥 Mittelwert über alle Fotos
        mean_embedding = np.mean(np.vstack(embeddings), axis=0)
        horse_vectors.append(mean_embedding)
        horse_names.append(horse_name)

    if not horse_vectors:
        print("⚠️ Keine Pferdebilder gefunden – Index nicht gebaut")
        return

    # FAISS Index bauen
    vectors = np.vstack(horse_vectors).astype("float32")
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "w") as f:
        json.dump(horse_names, f, indent=2)

    print("✅ Index mit 1 Vektor pro Pferd erstellt")


build_index()
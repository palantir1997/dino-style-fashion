import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.dino_model import DinoModel
from utils.chroma_utils import create_collection
from database.mongo_db import save_image_to_mongo
from PIL import Image
import chromadb
from chromadb.errors import NotFoundError

REFERENCE_DIR = "data/reference_images"

# 최신 방식: ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db") # 더 이상 Settings 필요 없음
# client = chromadb.Client()도 가능하지만 persist는 PersistentClient로

# 컬렉션 가져오기 또는 생성
try:
    collection = client.get_collection("style_collection")
except NotFoundError:
    collection = create_collection(client, name="style_collection")

dino_model = DinoModel()

for filename in os.listdir(REFERENCE_DIR):
    if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(REFERENCE_DIR, filename)
    pil_img = Image.open(img_path).convert("RGB").resize((320, 240))

    # MongoDB 저장
    mongo_id = save_image_to_mongo(img_path)

    # DINO 임베딩 → ChromaDB 저장
    embedding = dino_model.get_embedding(pil_img)
    collection.add(
        ids=[filename],
        embeddings=[embedding],
        metadatas=[{"filename": filename, "mongo_id": mongo_id}]
    )
    print(f"Registered: {filename}, MongoDB ID: {mongo_id}")

print("All reference images registered in MongoDB + ChromaDB")

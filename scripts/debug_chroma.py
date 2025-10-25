import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.dino_model import DinoModel
from utils.chroma_utils import create_collection
from database.mongo_db import get_image_from_mongo
from PIL import Image
import chromadb
from chromadb.errors import NotFoundError

# -------------------------------
# 1. PersistentClient 생성
# -------------------------------
client = chromadb.PersistentClient(path="./chroma_db")  # 경로 통일

# -------------------------------
# 2. 컬렉션 가져오기 또는 생성
# -------------------------------
try:
    collection = client.get_collection("style_collection")
    print("Collection found!")
except NotFoundError:
    print("Collection not found, creating new one...")
    collection = create_collection(client, name="style_collection")
    print("Collection 'style_collection' created!")

print("ChromaDB Collections:", client.list_collections())

# -------------------------------
# 3️. 컬렉션 데이터 확인
# -------------------------------
total = collection.count()
print(f"총 데이터 수: {total}")

if total > 0:
    # 최신 ChromaDB API: peek() 위치 인자 사용
    sample_data = collection.peek(1)
    print("샘플 데이터:", sample_data)
else:
    print("컬렉션에 임베딩 데이터가 없습니다. init_reference_images.py 먼저 실행하세요.")

# -------------------------------
# 4️. 쿼리 테스트 (샘플 이미지)
# -------------------------------
dino_model = DinoModel()
TEST_IMG_PATH = "data/reference_images/35806.jpg"  # 테스트용 이미지

if os.path.exists(TEST_IMG_PATH):
    pil_img = Image.open(TEST_IMG_PATH).convert("RGB").resize((320, 240))
    query_embedding = dino_model.get_embedding(pil_img)

    query_result = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    print("Query 결과:")
    # metadatas / distances가 존재하는지 확인 후 출력
    if query_result['metadatas'] and query_result['distances']:
        for i, meta in enumerate(query_result['metadatas'][0]):
            mongo_id = meta['mongo_id']
            img_path = get_image_from_mongo(mongo_id)
            score = query_result['distances'][0][i]
            print(f"{i+1}. MongoID: {mongo_id}, 거리: {score}, 이미지 경로: {img_path}")
    else:
        print("쿼리 결과가 없습니다.")
else:
    print(f"테스트 이미지가 존재하지 않습니다: {TEST_IMG_PATH}")

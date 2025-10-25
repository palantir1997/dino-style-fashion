import pymongo
from bson.binary import Binary
from bson import ObjectId
from io import BytesIO
from PIL import Image
import os

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["style_db"]
collection = db["images"]

def save_image_to_mongo(img_path, size=(320, 240)):
    """이미지를 MongoDB에 저장하고 ObjectId 반환"""
    img = Image.open(img_path).convert("RGB").resize(size)
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    doc = {
        "filename": os.path.basename(img_path),
        "data": Binary(buffer.getvalue())
    }
    return str(collection.insert_one(doc).inserted_id)

def load_image_from_mongo(mongo_id):
    """MongoDB에서 ObjectId로 이미지 불러오기"""
    doc = collection.find_one({"_id": ObjectId(mongo_id)})
    if doc:
        return Image.open(BytesIO(doc["data"]))
    return None

import io  # 추가

def get_image_from_mongo(mongo_id):
    doc = collection.find_one({"_id": ObjectId(mongo_id)})
    if not doc:
        return None
    img_data = doc["data"]  # save_image_to_mongo에서 'data'로 저장했음
    return Image.open(io.BytesIO(img_data))

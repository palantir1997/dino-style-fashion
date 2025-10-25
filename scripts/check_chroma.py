import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.chroma_utils import create_collection
import chromadb
from chromadb.errors import NotFoundError

# PersistentClient 생성 (최신 ChromaDB 방식)
client = chromadb.PersistentClient(path="./chroma_db")

# 컬렉션 가져오기 또는 없으면 생성
try:
    collection = client.get_collection("style_collection")
    print("Collection found!")
except NotFoundError:
    print("Collection not found, creating new one...")
    collection = create_collection(client, name="style_collection")
    print("Collection 'style_collection' created!")

print("ChromaDB Collections:", client.list_collections())

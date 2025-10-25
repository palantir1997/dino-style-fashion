import chromadb
from chromadb.config import Settings

# utils/chroma_utils.py
import chromadb

def get_chroma_client(path="./chroma_db"):
    return chromadb.PersistentClient(path=path)


def create_collection(client, name="style_collection"):
    return client.create_collection(name=name)

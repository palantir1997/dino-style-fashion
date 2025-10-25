import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from flask import Flask, render_template, request
from utils.chroma_utils import create_collection
from models.dino_model import DinoModel
from database.mongo_db import get_image_from_mongo
from PIL import Image
import chromadb
from chromadb.errors import NotFoundError
import torch
from transformers import CLIPProcessor, CLIPModel
from transformers import BlipProcessor, BlipForConditionalGeneration

app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------
# ChromaDB 연결 (DINO용)
# -----------------------------
client = chromadb.PersistentClient(path="./chroma_db")
try:
    dino_collection = client.get_collection("style_collection")
except NotFoundError:
    dino_collection = create_collection(client, name="style_collection")

# -----------------------------
# ChromaDB 연결 (CLIP용)
# -----------------------------
try:
    clip_collection = client.get_collection("clip_collection")
except NotFoundError:
    clip_collection = create_collection(client, name="clip_collection")

# -----------------------------
# DINO 모델 초기화
# -----------------------------
dino_model = DinoModel()

# -----------------------------
# CLIP 모델 초기화
# -----------------------------
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_clip_embedding(pil_img):
    inputs = clip_processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = clip_model.get_image_features(**inputs)
    return outputs.cpu().numpy().flatten().tolist()

# -----------------------------
# BLIP 모델 초기화 (업로드 이미지 캡션)
# -----------------------------
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
blip_model.eval()

def get_caption(pil_img):
    inputs = blip_processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption

# -----------------------------
# Flask 라우팅
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    uploaded_url = None
    results = []
    model_selected = "dino"
    description = ""  # 업로드 이미지 캡션

    if request.method == "POST":
        model_selected = request.form.get("model", "dino")

        if "file" not in request.files:
            return "No file uploaded", 400
        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400

        # 업로드된 파일 저장
        upload_path = os.path.join("static/uploads", file.filename)
        os.makedirs("static/uploads", exist_ok=True)
        file.save(upload_path)
        uploaded_url = "/" + upload_path

        # PIL 이미지 생성
        pil_img = Image.open(upload_path).convert("RGB").resize((320, 240))

        # -----------------------------
        # 업로드 이미지 캡션 생성
        # -----------------------------
        description = get_caption(pil_img)

        # -----------------------------
        # 모델별 처리
        # -----------------------------
        if model_selected == "dino":
            query_embedding = dino_model.get_embedding(pil_img)
            query_result = dino_collection.query(query_embeddings=[query_embedding], n_results=5)

            os.makedirs("static/recommend", exist_ok=True)
            for i, metadata in enumerate(query_result['metadatas'][0]):
                mongo_id = metadata["mongo_id"]
                pil_rec = get_image_from_mongo(mongo_id)
                if pil_rec is None:
                    continue
                rec_path = f"static/recommend/{mongo_id}.jpg"
                pil_rec.save(rec_path)
                results.append({
                    "path": "/" + rec_path,
                    "score": query_result['distances'][0][i],
                    "description": None
                })

        elif model_selected == "clip":
            query_embedding = get_clip_embedding(pil_img)
            query_result = clip_collection.query(query_embeddings=[query_embedding], n_results=5)

            os.makedirs("static/recommend", exist_ok=True)
            for i, metadata in enumerate(query_result['metadatas'][0]):
                mongo_id = metadata["mongo_id"]
                pil_rec = get_image_from_mongo(mongo_id)
                if pil_rec is None:
                    continue
                rec_path = f"static/recommend/{mongo_id}.jpg"
                pil_rec.save(rec_path)

                results.append({
                    "path": "/" + rec_path,
                    "score": query_result['distances'][0][i],
                    "description": None  # 추천 이미지 캡션은 따로 생성 안함
                })

    return render_template(
        "index.html",
        uploaded=uploaded_url,
        results=results,
        model_selected=model_selected,
        description=description
    )

if __name__ == "__main__":
    app.run(debug=True)

from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image

class ClipModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def get_embedding_and_description(self, pil_img):
        # 이미지 임베딩
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
        embedding = outputs.cpu().numpy().flatten().tolist()

        # 텍스트 설명은 간단히 이미지 이름/경로 등으로 표시 가능
        description = f"CLIP 임베딩 길이: {len(embedding)}"

        return embedding, description

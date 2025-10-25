from transformers import AutoImageProcessor, AutoModel
import torch

# DINO 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base', use_fast=False)
image_encoder = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
image_encoder.eval()

def get_dino_embedding(pil_img):
    """PIL 이미지를 받아서 DINO feature 벡터 반환"""
    inputs = image_processor(images=pil_img, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = image_encoder(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
    return embedding.tolist()

class DinoModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.model = AutoModel.from_pretrained('facebook/dinov2-base').to(self.device)
        self.model.eval()

    def get_embedding(self, pil_img):
        inputs = self.processor(images=pil_img, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        return embedding.tolist()

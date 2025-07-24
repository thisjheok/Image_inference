# inference.py

import torch
from torchvision import models, transforms
from PIL import Image
import requests

# 전처리 정의
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 모델 로딩 (최초 1회만 실행)
model = models.resnet18(pretrained=True)
model.eval()

# 클래스 레이블 로딩 (최초 1회만 실행)
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = requests.get(LABELS_URL).text.strip().split("\n")

def predict_image(image: Image.Image) -> dict:
    """PIL 이미지를 받아서 예측 결과 반환"""
    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_catid = torch.topk(probabilities, 1)

    return {
        "label": labels[top_catid.item()],
        "probability": round(top_prob.item(), 4)
    }

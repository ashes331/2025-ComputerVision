from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch 
import matplotlib.pyplot as plt

# 1. 이미지 불러오기
imgs = [
        Image.open('BSDS_242078.jpg'),
        Image.open('BSDS_361010.jpg'),
        Image.open('BSDS_376001.jpg')
]

# 2. ViT feature extractor + model
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

# 3. 이미지 전처리
inputs = processor(images=imgs, return_tensors="pt")

# 4. 모델 추론
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probs = logits.softmax(dim=1)

# 5. 결과 출력
for i in range(len(imgs)):
    plt.imshow(imgs[i]); plt.axis("off"); plt.show()
    
    pred_idx = int(torch.argmax(probs[i]))
    pred_label = model.config.id2label[pred_idx]
    pred_prob = float(probs[i][pred_idx] * 100)
    
    print(f"{i}번째 이미지 예측")
    print(f" ▶ 클래스: {pred_label}")
    print(f" ▶ 확률: {pred_prob:.2f}%\n")












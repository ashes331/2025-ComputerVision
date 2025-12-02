# python 3.9 버전 사용
# 가상환경을 파이썬 3.9로 만들어서 사용하기
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
!pip install transformers timm opencv-pyhon numpy==1.26.4

from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch
import numpy as np
import cv2 as cv

img=Image.open("BSDS_361010.jpg")

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

inputs = processor(images=imgs, return_tensors="pt")
outputs = model(**inputs)

for score, label, box in zip(results["score"], results["labels"], results["boxes"]):
    x_min, y_min, x_max, y_max = map(int, box.tolost())
    color = colors[label.item() % 100]
    name = model.config.id2label[label.item()]
    cv.rectangle(im, (x_min, y_min), (x_max, y_max), color, 2)
    cv.putText(im, f"{name} {score.item():.2f}", (x_min, y_min - 5),
                 cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
cv.imshow('DETR',im)
cv.waitKey()       
cv.destroyAllWindows()




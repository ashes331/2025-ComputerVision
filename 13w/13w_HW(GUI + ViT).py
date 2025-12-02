import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget
import cv2 as cv
import numpy as np
import torch
from transformers import ViTImageProcessor, ViTForImageClassification

class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('이미지 모델 추론 - 20241491 김성원')
        self.setGeometry(100, 100, 400, 170)

        self.layout = QVBoxLayout()

        self.image_label = QLabel(self)
        self.layout.addWidget(self.image_label)

        fileButton = QPushButton('이미지 불러오기', self)
        fileButton.setGeometry(10, 10, 100, 50)
        fileButton.clicked.connect(self.load_image)
        self.layout.addWidget(fileButton)

        predictButton = QPushButton('모델 추론하기', self)
        predictButton.setGeometry(120, 10, 100, 50)
        predictButton.clicked.connect(self.run_inference)
        self.layout.addWidget(predictButton)

        self.result_label = QLabel(self)
        self.layout.addWidget(self.result_label)

        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

        self.img = None
        self.mask = None

        self.processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, '이미지 불러오기', '', '이미지 파일 (*.jpg *.jpeg *.png)')
        if fname:
            self.img = cv.imread(fname)
            self.mask = np.zeros(self.img.shape[:2], np.uint8) 
            self.display_image(self.img)

    def display_image(self, img):
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        height, width, channels = img.shape
        bytes_per_line = channels * width
        q_img = img.tobytes()  

        self.image_label.setPixmap(self.create_pixmap_from_bytes(q_img, width, height, bytes_per_line))

    def create_pixmap_from_bytes(self, q_img, width, height, bytes_per_line):
        from PyQt5.QtGui import QImage, QPixmap

        image = QImage(q_img, width, height, bytes_per_line, QImage.Format_RGB888)
        return QPixmap(image)

    def run_inference(self):
        if self.img is not None:
            # 1. 이미지 전처리
            inputs = self.processor(images=self.img, return_tensors="pt")

            # 2. 모델 추론 
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = logits.softmax(dim=1)

            pred_idx = torch.argmax(probs, dim=1).item()
            pred_label = self.model.config.id2label[pred_idx]
            pred_prob = probs[0][pred_idx].item() * 100  
            
            # 4. 결과 출력
            self.result_label.setText(f" ▶ 클래스: {pred_label}\n ▶ 확률: {pred_prob:.2f}%\n")

        else:
            self.result_label.setText("이미지를 먼저 불러오세요.")

if __name__ == "__main__":
    app = QApplication(sys.argv)  
    window = ImageProcessingApp() 
    window.show()  
    sys.exit(app.exec_())  
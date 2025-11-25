import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

class ImageViewer(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()
        self.model = load_model('F:\Machine Diploma\\material\\project\\model.h5')  

    def initUI(self):
        self.setWindowTitle('Image Viewer')

        self.layout = QVBoxLayout()

        self.image_label_original = QLabel(self)
        self.image_label_original.setFixedSize(400, 300)

        self.image_label_canny = QLabel(self)
        self.image_label_canny.setFixedSize(400, 300)

        self.image_label_median = QLabel(self)
        self.image_label_median.setFixedSize(400, 300)

        self.image_label_otsu = QLabel(self)
        self.image_label_otsu.setFixedSize(400, 300)

        self.image_label_harris = QLabel(self)
        self.image_label_harris.setFixedSize(400, 300)

        self.image_label_sift = QLabel(self)
        self.image_label_sift.setFixedSize(400, 300)

        self.button_open = QPushButton('Open Image', self)
        self.button_open.clicked.connect(self.openFileDialog)

        self.button_canny = QPushButton('Apply Canny', self)
        self.button_canny.clicked.connect(self.applyCanny)

        self.button_median = QPushButton('Apply Median Filter', self)
        self.button_median.clicked.connect(self.applyMedian)

        self.button_otsu = QPushButton('Apply Otsu Threshold', self)
        self.button_otsu.clicked.connect(self.applyOtsu)

        self.button_harris = QPushButton('Apply Harris Corner', self)
        self.button_harris.clicked.connect(self.applyHarris)

        self.button_sift = QPushButton('Apply SIFT', self)
        self.button_sift.clicked.connect(self.applySift)

        self.button_classify = QPushButton('Classify Image', self)
        self.button_classify.clicked.connect(self.classifyImage)  

        self.result_label = QLabel('Result:', self)

        hbox1 = QHBoxLayout()
        hbox1.addWidget(self.image_label_original)
        hbox1.addWidget(self.image_label_canny)
        hbox1.addWidget(self.image_label_median)

        hbox2 = QHBoxLayout()
        hbox2.addWidget(self.image_label_otsu)
        hbox2.addWidget(self.image_label_harris)
        hbox2.addWidget(self.image_label_sift)

        self.layout.addLayout(hbox1)
        self.layout.addLayout(hbox2)
        self.layout.addWidget(self.button_open)
        self.layout.addWidget(self.button_canny)
        self.layout.addWidget(self.button_median)
        self.layout.addWidget(self.button_otsu)
        self.layout.addWidget(self.button_harris)
        self.layout.addWidget(self.button_sift)
        self.layout.addWidget(self.button_classify)  
        self.layout.addWidget(self.result_label)  

        self.setLayout(self.layout)

        self.img = None

    def openFileDialog(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image File", "",
                                                  "Images (*.png *.xpm *.jpg *.jpeg *.bmp *.gif);;All Files (*)", options=options)
        if fileName:
            self.img = cv2.imread(fileName)
            self.displayImage(self.img, self.image_label_original)

    def displayImage(self, img, label):
        height, width, channel = img.shape
        bytesPerLine = 3 * width
        qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        label.setPixmap(QPixmap.fromImage(qImg))

    def applyCanny(self):
        if self.img is not None:
            img_median = cv2.medianBlur(self.img, 5)
            img_canny = cv2.Canny(img_median, 100, 200)
            img_canny_colored = cv2.cvtColor(img_canny, cv2.COLOR_GRAY2BGR)
            self.displayImage(img_canny_colored, self.image_label_canny)

    def applyMedian(self):
        if self.img is not None:
            img_median = cv2.medianBlur(self.img, 5)
            self.displayImage(img_median, self.image_label_median)

    def applyOtsu(self):
        if self.img is not None:
            gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            _, img_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img_otsu_colored = cv2.cvtColor(img_otsu, cv2.COLOR_GRAY2BGR)
            self.displayImage(img_otsu_colored, self.image_label_otsu)

    def applyHarris(self):
        if self.img is not None:
            gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            gray = np.float32(gray)
            dst = cv2.cornerHarris(gray, 2, 3, 0.04)
            dst = cv2.dilate(dst, None)
            img_harris = self.img.copy()
            img_harris[dst > 0.01 * dst.max()] = [0, 0, 255]
            self.displayImage(img_harris, self.image_label_harris)

    def applySift(self):
        if self.img is not None:
            gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            img_sift = cv2.drawKeypoints(self.img, keypoints, None)
            self.displayImage(img_sift, self.image_label_sift)

    def classifyImage(self):
        if self.img is not None:
            image = cv2.resize(self.img, (100, 100))
            image = np.expand_dims(image, axis=0)

            prediction = self.model.predict(image)
            category = np.argmax(prediction, axis=1)

            self.result_label.setText(f'Result: {get_code(category[0])}')  

def get_code(n):
    code = {0: 'buildings', 1: 'forest', 2: 'glacier', 3: 'mountain', 4: 'sea', 5: 'street'}
    return code.get(n, 'Unknown')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())

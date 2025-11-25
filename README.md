# PyQt5 Image Viewer with Image Processing & Classification

This project is a **desktop application** built with **PyQt5** that allows users to open images, apply various **image processing techniques**, and classify images using a **pre-trained CNN model**.

---

## 🛠 Features

- Open images from your computer.  
- Apply image processing techniques:  
  - **Canny Edge Detection**  
  - **Median Filter**  
  - **Otsu Thresholding**  
  - **Harris Corner Detection**  
  - **SIFT Keypoints Detection**  
- Classify images into one of six categories:  
  - Buildings, Forest, Glacier, Mountain, Sea, Street  
- Display processed images side-by-side with the original image.  

---

## 📷 Screenshots

You can include screenshots of the application here:

![Original Image](assets/original_example.png)  
![Canny Example](assets/canny_example.png)  
![Classification Result](assets/result_example.png)  

---

## ⚙️ Requirements

- Python 3.8+  
- PyQt5  
- OpenCV (`opencv-python`)  
- TensorFlow / Keras  
- NumPy  

Install dependencies with:

```bash
pip install pyqt5 opencv-python tensorflow numpy

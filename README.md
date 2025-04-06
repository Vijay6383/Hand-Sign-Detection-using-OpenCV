# Real-Time Hand Sign Detection App using OpenCv, RandomForest and Streamlit

This project is a real-time **hand sign recognition system** that uses a **Random Forest classifier** trained on custom hand gesture images (0-9). It uses **OpenCV** and **MediaPipe** for hand landmark detection and **HOG (Histogram of Oriented Gradients)** for feature extraction. The model is deployed using **Streamlit**, providing a simple, webcam-based interface to recognize hand signs and display the corresponding digit.

## Hand Sign (0-9)

![App Screenshot](https://github.com/Vijay6383/Hand-Sign-Detection-using-OpenCV/blob/main/ASL_numbers_handsign.png)

## Data Collection Screenshot

![data collection](https://github.com/Vijay6383/Hand-Sign-Detection-using-OpenCV/blob/main/screenshot.jpg)

## Demo

[![App Demo video](https://github.com/Vijay6383/Hand-Sign-Detection-using-OpenCV/blob/main/Demo_video.jpg)](https://youtu.be/JsHrH0lhYH4?si=NrffCzWEcUaJ7MDs)

## Requirements
Make sure you have the following dependencies installed:

- **Mediapipe**: `0.10.5`
- **OpenCV (cv2)**: `4.7.0.72`
- **numpy**: `2.0.2`
- **scikit-image**: `0.24.0`
- **scikit-learn**: `1.6.1`
- **streamlit**: `1.44.0`

You can install them via pip:

```bash
pip install -r requirements.txt
```

## Dataset
I collected the dataset using my webcam, generating 100 images for each class (numbers 0-9) using the [`data_collection.py`](https://github.com/Vijay6383/Hand-Sign-Detection-using-OpenCV/blob/main/data_collection.py) module.

One of the main challenges in creating the dataset was ensuring accurate predictions for different hand orientations, distances from the webcam, and positions in the frame. I handled this challenge by capturing diverse images in various conditions, but the model's accuracy can still improve with more frames at different distances and positions.

The complete dataset is not uploaded here, but you can collect your own dataset using the aforementioned modules.

## 📌 Features

- Real-time hand detection using webcam
- HOG-based feature extraction from hand landmarks
- Multi-class classification (digits 0 to 9)
- Trained using a **Random Forest** model
- Easy-to-use **Streamlit web app**
- Custom dataset support (folder-wise image organization)

## 🧠 Model Overview

- **Model**: Random Forest Classifier
- **Features**: HOG features from cropped hand images
- **Training Data**: 10 folders (0-9), each containing ~100 images of hand signs
- **Preprocessing**: 
  - Hand landmark detection via MediaPipe
  - Cropping hand region
  - Resizing + HOG feature extraction
- **Accuracy**: ~ 98.5% on test data set

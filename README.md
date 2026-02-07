# AI-Medical-Diagnosis-CNN
A Deep Learning model using CNNs to classify brain MRI scans (Tumor vs Healthy) with 99% accuracy. Built with TensorFlow &amp; Keras.

# 🧠 AI Brain Tumor Classifier

An automated medical diagnosis system built with **Deep Learning (CNN)** to detect brain tumors from MRI scans.

## 🚀 Key Features
- **Synthetic Data Generation:** Created a custom dataset generator using OpenCV to simulate MRI scans.
- **CNN Architecture:** Built a Convolutional Neural Network with TensorFlow/Keras achieving **99% accuracy**.
- **Real-time Prediction:** Includes a testing script to diagnose new unseen images.

## 🛠️ Tech Stack
- **Python**
- **TensorFlow & Keras**
- **OpenCV** (Computer Vision)
- **NumPy**

## 📊 How It Works
1. `create_data.py`: Generates synthetic MRI images (Tumor/Healthy).
2. `brain_model.py`: Trains the CNN model on the dataset.
3. `test_model.py`: Loads the trained model and predicts results for random samples.

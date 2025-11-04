# 🤟 Sign Language Detection using Deep Learning

This project detects *American Sign Language (ASL)* hand gestures in real time using a *Convolutional Neural Network (CNN)*.  
It can recognize *A–Z alphabets* or *29 classes*.

A simple and intuitive *Streamlit web application* is included to enable real-time detection using your system's webcam.

---

## 📌 Features

✅ Real-time ASL hand gesture detection  
✅ Built using *TensorFlow/Keras* CNN model  
✅ Webcam integration via *OpenCV*  
✅ Interactive *Streamlit UI*  
✅ Easy to train, test, and deploy  

---

## 🗂 Dataset

We use the *ASL Alphabet Dataset* provided on Kaggle.

📥 *Download Link*: [ASL Alphabet Dataset – Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

- Total Images: *87,000+*
- Total Classes: *29*
  - 26 alphabets (A–Z)
  - 3 additional classes: space, del, and nothing

### 🧾 Dataset Setup Instructions:

1. Download and unzip the dataset.
2. Copy the asl_alphabet_train/ folder into the dataset/ directory of this project.

---

## ⚙ Installation

### 1️⃣ Clone this Repository


1)git clone https://github.com/<your-username>/sign-language-detection.git.

2)cd sign-language-detection.


### 2️⃣ Install Python 3.10

TensorFlow does not support Python 3.13+  
📥 [Download Python 3.10.11](https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe)

### 3️⃣ Create Virtual Environment


1)py -3.10 -m venv venv310.

2)venv310\Scripts\activate.


### 4️⃣ Install Required Packages


pip install --upgrade pip
pip install -r requirements.txt


Or manually install:


pip install tensorflow==2.12.0 opencv-python numpy matplotlib scikit-learn streamlit


---

## 🧠 CNN Model Architecture

The CNN model used for training consists of the following layers:  
Input: (64, 64, 3) RGB image

- Conv2D(32 filters, 3x3) → ReLU → MaxPooling(2x2)
- Conv2D(64 filters, 3x3) → ReLU → MaxPooling(2x2)
- Conv2D(128 filters, 3x3) → ReLU → MaxPooling(2x2)
- Flatten
- Dense(512) → ReLU → Dropout(0.5)
- Dense(29) → Softmax

- Loss Function: categorical_crossentropy
- Optimizer: adam
- Training Epochs: 10
- Validation Accuracy: ~95%

---

## 🚀 Training the Model

Run the training script:


python train_model.py


This will:
- Load the dataset from dataset/asl_alphabet_train/
- Train the CNN model
- Save the model as asl_model.h5 for later inference

---

## 🎥 Run the Streamlit App

To launch the real-time sign detection app:


streamlit run app.py


The app will:
- Load your webcam
- Use the trained CNN model
- Predict and display the detected sign in real time
- Press Ctrl + C in the terminal to stop the app.

---

## 📋 requirements.txt


tensorflow==2.12.0
opencv-python
numpy
matplotlib
scikit-learn
streamlit


---

## 🔮 Future Enhancements

- 🎤 Integrate Text-to-Speech (TTS) for full ASL-to-audio translation
- 🌍 Support Indian or regional sign languages
- 🖐 Add MediaPipe Hands for better hand segmentation
- 📈 Improve accuracy with more advanced CNN architectures or transfer learning

---

## 🙌 Acknowledgements

- 📊 Dataset: [ASL Alphabet Dataset by Grassknoted](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- 🔍 TensorFlow & Keras for deep learning
- 🎛 Streamlit for the app interface
- 📷 OpenCV for webcam functionality

---

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("asl_model.h5")

# Class labels A-Z
classes = [chr(i) for i in range(65, 91)]

st.title("🤟 ASL Sign Language Recognition")
st.write("Show a hand gesture (A-Z) and the model will predict it.")

# Start webcam
run = st.checkbox("Start Webcam")
FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)
IMG_SIZE = 64

while run:
    ret, frame = camera.read()
    if not ret:
        st.write("❌ Webcam not accessible")
        break

    # Resize & preprocess
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    preds = model.predict(img)
    pred_class = classes[np.argmax(preds)]
    prob = np.max(preds) * 100

    # Show prediction on frame
    cv2.putText(frame, f"{pred_class} ({prob:.2f}%)", (20,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

camera.release()

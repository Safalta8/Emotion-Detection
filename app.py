import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import pandas as pd
import altair as alt
import cv2

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Emotion Detection",
    page_icon="🎭",
    layout="wide"
)

# ---------------- MODERN UI THEME ----------------
st.markdown("""
<style>
/* Main background */
.stApp {
    background-color: #F0EEEA;  /* soft background */
}

/* Headings */
h1, h2, h3 {
    color: #3C2A21;  /* deep brown */
}

/* Card style */
.block-container {
    padding: 2rem;
    background-color: #D2E0D3;  /* light sage */
    border-radius: 12px;
}

/* Upload box */
[data-testid="stFileUploader"] {
    background-color: #F0EEEA;  /* soft neutral */
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #97B3AE;  /* soft green border */
}

/* Buttons */
.stButton>button {
    background-color: #F0EEEA;  /* soft neutral */
    color: #111827;
    border-radius: 8px;
    border: none;
    padding: 8px 16px;
}
.stButton>button:hover {
    background-color: #D6CBBF;  /* taupe hover */
}

/* Metric */
[data-testid="stMetricValue"] {
    color: #111827;
}

/* Image border */
img {
    border-radius: 10px;
    border: 2px solid #97B3AE;  /* soft green border */
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🎭 Emotion Detection App")

# ---------------- ABOUT ----------------
st.markdown("""
### 📌 About
This app showcases **Human Emotion Detection** using deep learning.  
Powered by a **Convolutional Neural Network (CNN)**, it can analyze facial images from uploaded photos to predict emotions such as **happiness, sadness, anger, and surprise**.  
""")

# ---------------- LOAD MODEL ----------------
model = load_model(r"C:\\Users\\Des\\Downloads\\archive (4)\\emotion_model.h5")

emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------- FUNCTION ----------------
def detect_emotions(img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    emotion_data = None

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))

        roi = roi_gray.astype('float32') / 255.0
        roi = np.reshape(roi, (1, 48, 48, 1))

        prediction = model.predict(roi, verbose=0)
        emotion_data = prediction.flatten()

        label = emotion_labels[np.argmax(prediction)]
        confidence = np.max(prediction)

        # Box color (soft green)
        cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (151,179,174), 2)

        cv2.putText(img_rgb, f"{label} ({confidence:.2f})",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (151,179,174),
                    2)

    return img_rgb, emotion_data, len(faces)

# ---------------- LAYOUT ----------------
col1, col2 = st.columns([2,1])

# ---------------- IMAGE UPLOAD ----------------
with col1:
    st.subheader("📂 Upload Image")

    uploaded_file = st.file_uploader("", type=["jpg","png","jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        img_rgb = np.array(image)

        result_img, emotion_data, face_count = detect_emotions(img_rgb)

        st.image(result_img, caption="Detected Emotion", width=600)

# ---------------- ANALYSIS ----------------
with col2:
    st.subheader("📊 Analysis")

    if uploaded_file is not None and emotion_data is not None:
        st.metric("👤 Faces Detected", face_count)

        df = pd.DataFrame({
            'Emotion': emotion_labels,
            'Confidence': emotion_data
        })

        # Light pastel color mapping
        color_map = {
            'Happy': '#97B3AE',
            'Sad': '#D2E0D3',
            'Angry': '#F0DDD6',
            'Surprise': '#F2C3B9',
            'Fear': '#D6CBBF',
            'Disgust': '#F0EEEA',
            'Neutral': '#C4B7A6'
        }

        chart = alt.Chart(df).mark_bar().encode(
            x='Emotion',
            y='Confidence',
            color=alt.Color('Emotion:N', scale=alt.Scale(
                domain=list(color_map.keys()),
                range=list(color_map.values())
            ))
        ).properties(
            width=400,
            height=400
        )

        st.altair_chart(chart, use_container_width=False)

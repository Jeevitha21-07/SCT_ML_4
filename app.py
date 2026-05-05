import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model("model/gesture_model.keras")
class_names = sorted(os.listdir("dataset/train"))

st.set_page_config(page_title="Gesture Recognition", layout="centered")

st.title("✋ Hand Gesture Recognition")
st.write("Choose Image Upload or Webcam Detection")

# ---------------- MODE SELECT ----------------
mode = st.radio("Select Mode:", ["📸 Image Upload", "🎥 Webcam"])

# =========================================================
# 📸 IMAGE UPLOAD MODE
# =========================================================
if mode == "📸 Image Upload":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image = image.resize((64, 64))

        st.image(image, caption="Uploaded Image", width=300)

        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        st.success(f"Prediction: {predicted_class}")
        st.info(f"Confidence: {confidence:.2f}")

# =========================================================
# 🎥 WEBCAM MODE
# =========================================================
elif mode == "🎥 Webcam":

    class GestureDetector(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")

            h, w, _ = img.shape
            size = 200
            x1 = w//2 - size//2
            y1 = h//2 - size//2
            x2 = x1 + size
            y2 = y1 + size

            roi = img[y1:y2, x1:x2]

            try:
                roi_resized = cv2.resize(roi, (64, 64))
                img_array = roi_resized / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                prediction = model.predict(img_array, verbose=0)
                label = class_names[np.argmax(prediction)]
                confidence = float(np.max(prediction))

                if confidence > 0.75:
                    text = f"{label} ({confidence:.2f})"
                    color = (0, 255, 0)
                else:
                    text = "Detecting..."
                    color = (0, 0, 255)

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            except:
                pass

            return img

    st.info("Click Start and allow camera access")

    webrtc_streamer(
        key="gesture",
        video_transformer_factory=GestureDetector
    )
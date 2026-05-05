import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import av

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model("model/gesture_model.keras")

# 🔥 IMPORTANT: manually define class names (NO dataset dependency)
class_names = [
    "down", "fist", "fist_moved", "l", "okay",
    "palm", "palm_moved", "peace", "rock", "stop"
]

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Hand Gesture Recognition", layout="centered")

st.title("✋ Hand Gesture Recognition")
st.write("Choose Image Upload or Webcam Detection")

mode = st.radio("Select Mode:", ["📷 Image Upload", "📹 Webcam"])

# =========================================================
# 📷 IMAGE UPLOAD MODE
# =========================================================
if mode == "📷 Image Upload":

    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image = image.resize((64, 64))

        st.image(image, caption="Uploaded Image", width=300)

        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        confidence = float(np.max(prediction))
        predicted_class = class_names[np.argmax(prediction)]

        if confidence < 0.75:
            st.warning("⚠️ Not confident prediction")
        else:
            st.success(f"Prediction: {predicted_class} ({confidence:.2f})")

# =========================================================
# 📹 WEBCAM MODE (REAL-TIME)
# =========================================================
else:
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

    st.write("Allow camera access 👇")

    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")

            # Resize for prediction
            img_resized = cv2.resize(img, (64, 64))
            img_array = img_resized / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            confidence = float(np.max(prediction))
            predicted_class = class_names[np.argmax(prediction)]

            # Show only if confident
            if confidence > 0.75:
                text = f"{predicted_class} ({confidence:.2f})"
                cv2.putText(img, text, (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)

            return img

    webrtc_streamer(key="gesture", video_transformer_factory=VideoTransformer)
#type: ignore
import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import os
from collections import deque

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model("model/gesture_model.keras")
class_names = sorted(os.listdir("dataset/train"))

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mp_draw = mp.solutions.drawing_utils

# ---------------- SMOOTHING ----------------
pred_queue = deque(maxlen=10)

# ---------------- WEBCAM ----------------
cap = cv2.VideoCapture(0)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            h, w, _ = frame.shape
            x_list, y_list = [], []

            for lm in hand_landmarks.landmark:
                x_list.append(int(lm.x * w))
                y_list.append(int(lm.y * h))

            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)

            # Padding
            padding = 40
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            # Make square box
            cx = (x_min + x_max) // 2
            cy = (y_min + y_max) // 2
            size = max(x_max - x_min, y_max - y_min)

            x_min = max(0, cx - size // 2)
            y_min = max(0, cy - size // 2)
            x_max = min(w, cx + size // 2)
            y_max = min(h, cy + size // 2)

            # Ignore very small detections
            if (x_max - x_min) < 80 or (y_max - y_min) < 80:
                continue

            # Crop hand
            hand_img = frame[y_min:y_max, x_min:x_max]

            if hand_img.size != 0:
                try:
                    hand_resized = cv2.resize(hand_img, (64, 64))
                    img_array = hand_resized / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    # Prediction
                    prediction = model.predict(img_array, verbose=0)
                    pred_queue.append(prediction)

                    avg_pred = np.mean(pred_queue, axis=0)

                    predicted_class = class_names[np.argmax(avg_pred)]
                    confidence = float(np.max(avg_pred))

                    # Confidence filtering
                    if confidence < 0.75:
                        text = "Detecting..."
                        color = (0, 0, 255)
                    else:
                        text = f"{predicted_class} ({confidence:.2f})"
                        color = (0, 255, 0)

                    # Draw box
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

                    # Draw text
                    cv2.putText(frame, text, (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                except:
                    pass

            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
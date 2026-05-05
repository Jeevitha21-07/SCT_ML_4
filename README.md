# ✋ Hand Gesture Recognition App

This project is a **Machine Learning-based Hand Gesture Recognition System** built using **TensorFlow and Streamlit**.
It can detect hand gestures from both **uploaded images** and **real-time webcam input**.

---

## 🚀 Features

* 📷 Image Upload Gesture Detection
* 📹 Real-time Webcam Gesture Recognition
* 🤖 Deep Learning Model (TensorFlow / Keras)
* 🎯 Confidence-based Prediction Filtering
* ⚡ Fast and Interactive UI using Streamlit

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* OpenCV
* Streamlit
* NumPy
* PIL

---

## 📂 Project Structure

```
hand-gesture-recognition/
│
├── app.py                  # Main Streamlit application
├── requirements.txt       # Dependencies
├── model/
│   └── gesture_model.keras
├── .gitignore
```

---

## ▶️ How to Run Locally

### 1. Clone the repository

```
git clone https://github.com/YOUR_USERNAME/SCT_ML_4.git
cd SCT_ML_4
```

### 2. Create virtual environment

```
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Run the app

```
streamlit run app.py
```

---

## 🧠 Model Details

* Input size: **64 x 64 RGB images**
* Model: **CNN (Convolutional Neural Network)**
* Output: Gesture classification with confidence score

---

## 🎯 Supported Gestures

* Down
* Fist
* Fist Moved
* Index
* Okay
* Palm
* Palm Moved
* Peace
* Rock
* Stop

---

## ⚠️ Note

This application is designed to run **locally** due to TensorFlow compatibility limitations with some cloud environments.

---

## 📸 Demo

<img width="165" height="177" alt="image" src="https://github.com/user-attachments/assets/9fb6cd73-c620-4cff-aff2-813dbfe04b5f" />
<img width="563" height="443" alt="image" src="https://github.com/user-attachments/assets/992ec354-d367-47e6-97d9-209053859b12" />

<img width="312" height="304" alt="image" src="https://github.com/user-attachments/assets/f144e7df-7dda-4155-9283-e9d135297aba" />
<img width="358" height="316" alt="image" src="https://github.com/user-attachments/assets/35d49cfc-62e1-4898-81e6-7a6831032de1" />
<img width="309" height="275" alt="image" src="https://github.com/user-attachments/assets/8b6a892d-ed96-4d48-87d5-9f31adec8a11" />

---

## 📌 Future Improvements

* Deploy using lightweight models (TFLite)
* Improve real-time webcam performance
* Add more gesture classes
* Enhance UI/UX

---

## 🙌 Acknowledgement

This project was developed as part of a **Machine Learning Internship Task**.

---

## 👩‍💻 Author

**Jeevitha S**
AIML Engineering Student

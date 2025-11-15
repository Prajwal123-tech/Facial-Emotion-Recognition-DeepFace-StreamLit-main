# Facial-Emotion-Recognition-DeepFace-StreamLit-main
A real-time Facial Emotion Recognition (FER) web application built using DeepFace and Streamlit. The app detects human facial expressions from images or live webcam input and classifies them into emotional states such as Happy, Sad, Angry, Neutral, Surprise, Fear, and Disgust.

🚀 Features

🔍 Real-Time Emotion Detection using DeepFace

🖼 Image Upload Support

📸 Live Webcam Emotion Recognition

📊 Emotion probability scores & visual results

🧠 Uses Deep Learning (DeepFace Framework)

🌐 Simple & interactive UI built with Streamlit

🧠 How It Works

User uploads an image or starts the webcam.

The DeepFace model detects faces and classifies the emotion.

The app overlays predictions and displays confidence scores.

Streamlit updates the results in real-time.

🛠 Tech Stack

Python

DeepFace

OpenCV

TensorFlow/Keras

Streamlit

NumPy / PIL

📁 Project Structure
Facial-Emotion-Recognition-DeepFace-StreamLit-main/
│── app.py                 # Main Streamlit application
│── requirements.txt       # Dependencies
│── README.md              # Project documentation
│── images/                # Sample images (optional)
└── utils/                 # Additional helper functions (optional)

📦 Installation
1️⃣ Clone the repository
git clone https://github.com/your-username/Facial-Emotion-Recognition-DeepFace-StreamLit-main.git
cd Facial-Emotion-Recognition-DeepFace-StreamLit-main

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the Streamlit App
streamlit run app.py

📸 Usage

🔹 Option 1: Use Live Webcam

Click on the “Use Webcam” option.

Emotion detection updates in real time.

📊 Example Output

Detected emotion: Happy

Confidence: 98%

Additional emotions with probabilities

⚙ Requirements

Add these to requirements.txt if not already included:
streamlit
deepface
opencv-python
tensorflow
numpy
pillow

🧩 Future Enhancements

🎭 Multi-face emotion detection

🕒 Real-time performance optimization

📊 Emotion trend tracking

🌍 Deploy on cloud (Heroku / Streamlit Cloud / AWS)

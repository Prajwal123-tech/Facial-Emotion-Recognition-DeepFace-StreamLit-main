# Facial-Emotion-Recognition-DeepFace-StreamLit-main
A real-time Facial Emotion Recognition (FER) web application built using DeepFace and Streamlit. The app detects human facial expressions from images or live webcam input and classifies them into emotional states such as Happy, Sad, Angry, Neutral, Surprise, Fear, and Disgust.
🚀 Features

🔍 Real-Time Emotion Detection using DeepFace

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
🔹 Option 1: Upload an Image

Upload any photo containing a face.

The app analyzes it and displays predicted emotions.

🔹 Option 2: Use Live Webcam

Click on the “Use Webcam” option.

Emotion detection updates in real time.

📊 Example Output

Detected emotion: Happy

Confidence: 98%

Additional emotions with probabilities

(App screenshot or sample image can be added here)

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

Code:-

import numpy as np
from deepface import DeepFace

# Function to analyze facial attributes using DeepFace
def analyze_frame(frame):
    result = DeepFace.analyze(img_path=frame, actions=['age', 'gender', 'race', 'emotion'],
                              enforce_detection=False,
                              detector_backend="opencv",
                              align=True,
                              silent=False)
    return result


def overlay_text_on_frame(frame, texts):
    overlay = frame.copy()
    alpha = 0.9  # Adjust the transparency of the overlay
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (255, 255, 255), -1)  # White rectangle
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, 0)

    text_position = 15 # Where the first text is put into the overlay
    for text in texts:
        cv2.putText(frame, text, (10, text_position), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        text_position += 20

    return frame
def facesentiment():
    cap = cv2.VideoCapture(0)
    stframe = st.image([])  # Placeholder for the webcam feed

    if not cap.isOpened():
        st.error("❌ Could not open webcam. Please ensure it's connected and not being used by another application.")
        return

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            st.warning("⚠️ Failed to grab frame. Skipping...")
            continue  # Skip to the next iteration

        try:
            # Analyze the frame using DeepFace
            result = analyze_frame(frame)

            # Extract the face coordinates
            face_coordinates = result[0]["region"]
            x, y, w, h = face_coordinates['x'], face_coordinates['y'], face_coordinates['w'], face_coordinates['h']

            # Draw bounding box around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"{result[0]['dominant_emotion']}"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            # Convert the BGR frame to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Overlay white rectangle with text on the frame
            texts = [
                f"Age: {result[0]['age']}",
                f"Face Confidence: {round(result[0]['face_confidence'],3)}",
                f"Gender: {result[0]['dominant_gender']} {round(result[0]['gender'][result[0]['dominant_gender']], 3)}",
                f"Race: {result[0]['dominant_race']}",
                f"Dominant Emotion: {result[0]['dominant_emotion']} {round(result[0]['emotion'][result[0]['dominant_emotion']], 1)}",
            ]

            frame_with_overlay = overlay_text_on_frame(frame_rgb, texts)

            # Display the frame in Streamlit
            stframe.image(frame_with_overlay, channels="RGB")

        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
            continue  # Skip the problematic frame and continue

    cap.release()
    cv2.destroyAllWindows()

def main():
    # Face Analysis Application #
    # st.title("Real Time Face Emotion Detection Application")
    activities = ["Webcam Face Detection", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)
    st.sidebar.markdown(
        """ Developed by Prajwal Tanksalimath VTU CGPS Kalaburagi    
            Email : prajwaltanksalimath8074@gmail.com 
        """)
    if choice == "Webcam Face Detection":
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Real time face emotion recognition of webcam feed using OpenCV, DeepFace and Streamlit.</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        facesentiment()

    elif choice == "About":
        st.subheader("About this app")

        html_temp4 = """
            <div style="background-color:#98AFC7;padding:10px;border-radius:10px;">
                <h3 style="color:white;text-align:center;">Real-Time Facial Emotion Detection Application</h3>
                <p style="color:white;text-align:center;">
                    This application is designed to detect and classify human facial emotions in real time using advanced computer vision technologies.
                    It integrates <b>OpenCV</b> for real-time camera input and facial detection, <b>DeepFace</b> for accurate emotion analysis using deep learning models,
                    and <b>Streamlit</b> for deploying an interactive and user-friendly web interface.
                </p>
                <p style="color:white;text-align:center;">
                    The project showcases the practical implementation of artificial intelligence in understanding human emotions and has potential applications in mental health analysis,
                    user experience enhancement, and security systems.
                </p>
                <hr style="border:1px solid white;width:100%;">
                <h4 style="color:white;text-align:center;">Developed by Prajwal Vtu</h4>
                <p style="color:white;text-align:center;">
                    Prajwal is a final year MCA student at VTU Gulbarga. This is his final semester major academic project.
                    The development of this project involved significant effort and dedication. His internal guide played a key role in supporting and mentoring him throughout the process.
                </p>
                <h4 style="color:white;text-align:center;">Thank you for visiting!</h4>
            </div>
            <br><br>
        """

        st.markdown(html_temp4, unsafe_allow_html=True)

    else:
        pass

if __name__ == "__main__":
    main()

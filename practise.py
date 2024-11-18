import cv2
import numpy as np
from keras.models import load_model
from playsound import playsound
import threading
from PIL import Image
import streamlit as st

# Load the pre-trained model and face cascade
model = load_model('C:\\Project2\\DDS\\Scripts\\drowsiness.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Alert function to play sound
def alert():
    playsound('C:\\Project2\\DDS\\Scripts\\music.wav')

# Drowsiness detection function
def detect_drowsiness(source, stop_event):
    video_capture = cv2.VideoCapture(source)

    if not video_capture.isOpened():
        print("Error: Could not open video stream.")
        return

    closed_eye_counter = 0
    drowsiness_threshold = 15
    alert_active = False
    EAR_THRESHOLD = 0.25

    while not stop_event.is_set():
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture frame from video source.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_color = frame[y:y + h, x:x + w]
            roi_color = cv2.resize(roi_color, (224, 224))
            roi_color = roi_color.astype('float32') / 255.0
            roi_color = np.reshape(roi_color, (1, 224, 224, 3))

            prediction = model.predict(roi_color)

            if prediction[0][0] > EAR_THRESHOLD:
                closed_eye_counter += 1
                if closed_eye_counter > drowsiness_threshold and not alert_active:
                    alert()
                    alert_active = True
                cv2.putText(frame, "Driver Drowsy", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                closed_eye_counter = 0
                alert_active = False
                cv2.putText(frame, "Awake", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Convert the frame to an image that Streamlit can display
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Display the frame in Streamlit
        st.image(img_pil, use_column_width=True)

    video_capture.release()

# Streamlit UI for Streaming
st.title("🚗 Driver Drowsiness Detector 🚨")
choice = st.sidebar.selectbox("Navigation", ("Home","Tutorial","IP Camera", "Web Camera"))

# Control for IP Camera
if choice == "IP Camera":
    st.subheader("IP Camera Streaming")
    ip_camera_url = st.text_input("Enter IP Camera URL:", "http://10.10.10.103:8080/video")

    if st.button("Start Streaming"):
        if ip_camera_url:
            stop_event = threading.Event()
            streaming_thread = threading.Thread(target=detect_drowsiness, args=(ip_camera_url, stop_event))
            streaming_thread.start()

    if st.button("Stop Streaming"):
        stop_event.set()

# Control for Webcam
elif choice == "Web Camera":
    st.subheader("Web Camera Stream")

    if st.button("Start Webcam Streaming"):
        stop_event = threading.Event()
        streaming_thread = threading.Thread(target=detect_drowsiness, args=(0, stop_event))  # 0 for the default webcam
        streaming_thread.start()

    if st.button("Stop Webcam Streaming"):
        stop_event.set()

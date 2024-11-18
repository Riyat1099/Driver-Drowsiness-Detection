import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from playsound import playsound
from PIL import Image

# Loading the pre-trained model and face cascade
model = load_model('C:\\Project2\\DDS\\Scripts\\drowsiness.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def alert():
    playsound('C:\\Project2\\DDS\\Scripts\\music.wav')
def detect_drowsiness(source):
    video_capture = cv2.VideoCapture(source)

    if not video_capture.isOpened():
        st.error("Error: Could not open video stream.")
        return

    closed_eye_counter = 0
    drowsiness_threshold = 15
    alert_active = False
    EAR_THRESHOLD = 0.25

    frame_placeholder = st.empty()  

    while True:
        ret, frame = video_capture.read()
        if not ret:
            st.error("Failed to capture frame from video source.")
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

        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        frame_placeholder.image(img_pil, use_column_width=True)

    video_capture.release()

#Page Config:
st.set_page_config(page_title="DDD", page_icon="🚗")
st.title("🚗 Driver Drowsiness Detector 🚨")
choice = st.sidebar.selectbox("Navigation", ("Home","Tutorial","IP Camera", "Web Camera"))

if choice == "Home":
    st.image("https://www.ai-tech.systems/wp-content/uploads/2021/07/Driver-Drowsiness-Detection-using-CNN.gif")
    st.markdown(
        "<h2 style='background-color:skyblue; color:white; padding:10px; border-radius:5px; text-align:center;'>Welcome to SkyBridge Cargo!!</h2>",
        unsafe_allow_html=True
    )

elif choice == "Tutorial":
    st.subheader("Video Tutorial")
    st.video("Tutorial.mp4")  
    st.write("This is a Tutorial video of the Driver Drowsiness Detection System in action.")


elif choice == "IP Camera":
    st.subheader("IP Camera Streaming")
    ip_camera_url = st.text_input("Enter IP Camera URL:", "http://10.10.10.103:8080/video")

    if st.button("Start Streaming"):
        if ip_camera_url:
            detect_drowsiness(ip_camera_url)

    if st.button("Stop Streaming"):
        st.stop()  
elif choice == "Web Camera":
    st.subheader("Web Camera Stream")

    if st.button("Start Webcam Streaming"):
        detect_drowsiness(0)  

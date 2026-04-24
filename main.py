import os
import time
import logging
import tempfile
import cv2 as cv
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# Path to the best trained model
MODEL_DIR = './runs/detect/train/weights/best.pt'

logging.basicConfig(
    filename="./logs/log.log", 
    filemode='a', 
    level=logging.INFO, 
    format='%(asctime)s:%(levelname)s:%(name)s:%(message)s'
)

def main():
    # Load the YOLO model
    global model
    model = YOLO(MODEL_DIR)

    # Sidebar with animal class list
    st.sidebar.header("**Animal Classes**")
    class_names = ['Buffalo', 'Elephant', 'Rhino', 'Zebra', "Cheetah", "Fox", "Jaguar", "Tiger", "Lion", "Panda"]
    for animal in class_names:
        st.sidebar.markdown(f"- *{animal.capitalize()}*")

    # Title and description
    st.title("Real-time Animal Species Detection")
    st.write("The aim of this project is to develop an efficient computer vision model capable of real-time wildlife detection.")

    # Options for input: Image, Video, Webcam
    input_type = st.selectbox("Select Input Type", ["Upload Image", "Upload Video", "Use Webcam"])

    if input_type == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            inference_images(uploaded_file)

    elif input_type == "Upload Video":
        uploaded_file = st.file_uploader("Upload a video", type=['mp4', 'mov', 'avi'])
        if uploaded_file:
            inference_video(uploaded_file)

    elif input_type == "Use Webcam":
        st.write("Click on the button below to start webcam feed.")
        webcam_button = st.button("Start Webcam")
        if webcam_button:
            inference_webcam()


def inference_images(uploaded_file):
    image = Image.open(uploaded_file)
    # Predict the image using the model
    predict = model.predict(image)
    boxes = predict[0].boxes
    plotted = predict[0].plot()[:, :, ::-1]

    if len(boxes) == 0:
        st.markdown("**No Detection**")

    # Display the result image
    st.image(plotted, caption="Detected Image", width=600)
    logging.info("Detected Image")


def inference_video(uploaded_file):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    temp_file.close()

    cap = cv.VideoCapture(temp_file.name)
    frame_count = 0
    if not cap.isOpened():
        st.error("Error opening video file.")
    
    frame_placeholder = st.empty()
    stop_placeholder = st.button("Stop")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1
        if frame_count % 2 == 0:
            # Predict the frame
            predict = model.predict(frame, conf=0.75)
            # Plot boxes on the frame
            plotted = predict[0].plot()

            # Display the video frame
            frame_placeholder.image(plotted, channels="BGR", caption="Video Frame")
        
        # Clean up the temporary file
        if stop_placeholder:
            os.unlink(temp_file.name)
            break

    cap.release()  
    

def inference_webcam():
    cap = cv.VideoCapture(0)
    frame_count = 0

    if not cap.isOpened():
        st.error("Error accessing webcam.")
    
    frame_placeholder = st.empty()
    stop_placeholder = st.button("Stop Webcam")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1
        if frame_count % 2 == 0:
            # Predict the frame
            predict = model.predict(frame, conf=0.75)
            # Plot boxes on the frame
            plotted = predict[0].plot()

            # Display the video frame
            frame_placeholder.image(plotted, channels="BGR", caption="Webcam Feed")

        if stop_placeholder:
            break

    cap.release()
    

if __name__ == '__main__':
    main()

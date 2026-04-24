import os
import cv2
import tempfile
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Path to the trained YOLOv8 model
MODEL_PATH = "runs/detect/train/weights/best.pt"

# Initialize Streamlit UI
st.set_page_config(page_title="Animal Detection Webcam", layout="wide")
st.title("🐾 Real-Time Animal Species Detection")
st.write("This app uses a YOLOv8 custom-trained model to detect animals via Image, Video, or Webcam stream.")

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Sidebar Setup
st.sidebar.title("Configuration")
input_option = st.sidebar.radio("Select Input Mode:", ["Use Webcam", "Upload Image", "Upload Video"])

if input_option == "Use Webcam":
    st.header("Live Webcam Detection")
    st.write("Click 'Start Webcam' to begin detecting animals via your camera.")
    
    start_btn = st.button("Start Webcam")
    stop_btn = st.button("Stop Webcam")
    
    if start_btn:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Error accessing webcam. Please check your camera permissions.")
        else:
            frame_placeholder = st.empty()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read frame from webcam.")
                    break
                
                # YOLOv8 Inference
                results = model(frame, conf=0.5)
                
                # Plot the bounding boxes on the frame
                annotated_frame = results[0].plot()
                
                # Display the annotated frame in Streamlit
                frame_placeholder.image(annotated_frame, channels="BGR")
                
                if stop_btn:
                    break
            
            cap.release()

elif input_option == "Upload Image":
    st.header("Image Detection")
    uploaded_file = st.file_uploader("Upload an Image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Detecting...")
        
        # YOLOv8 Inference
        results = model(image, conf=0.5)
        annotated_image = results[0].plot()
        
        st.image(annotated_image, caption="Detected Result", use_column_width=True)

elif input_option == "Upload Video":
    st.header("Video Detection")
    uploaded_file = st.file_uploader("Upload a Video...", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        # Save temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())
        temp_file.close()
        
        cap = cv2.VideoCapture(temp_file.name)
        frame_placeholder = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # YOLOv8 Inference
            results = model(frame, conf=0.5)
            annotated_frame = results[0].plot()
            
            frame_placeholder.image(annotated_frame, channels="BGR")
            
        cap.release()
        os.unlink(temp_file.name)

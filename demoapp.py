import streamlit as st
import cv2
import torch
import sqlite3
import datetime
import geocoder
import numpy as np

# Fix for PosixPath issue on Windows
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Initialize the SQLite database
def init_db():
    conn = sqlite3.connect('object_detection.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS detections (id INTEGER PRIMARY KEY AUTOINCREMENT, object_name TEXT, count INTEGER, timestamp TEXT, location TEXT)''')
    conn.commit()
    return conn, c

# Insert detection data into the database
def log_detection(c, object_name, count, location):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO detections (object_name, count, timestamp, location) VALUES (?, ?, ?, ?)", (object_name, count, timestamp, location))
    c.connection.commit()

# Load YOLOv5 model
def load_model(model_file):
    temp_model_path = 'temp_model.pt'
    with open(temp_model_path, 'wb') as f:
        f.write(model_file.read())
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=temp_model_path, force_reload=True)
        model.eval()  # Set model to evaluation mode
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Get location using geocoder
def get_location():
    g = geocoder.ip('me')
    return f"{g.city}, {g.country}"

# Streamlit interface
st.title("SG Labs Real-Time Tracking of Adrian Steel Items")
model_path = st.file_uploader("Upload your YOLOv5 model (.pt)", type=['pt'])

if model_path:
    model = load_model(model_path)
    if model is not None:
        st.success("Model loaded successfully!")
        conn, c = init_db()
        location = get_location()
        st.text(f"Location: {location}")

        # Select camera index
        camera_index = st.sidebar.number_input("Select Camera Index", min_value=0, max_value=10, value=0)

        # Start camera button
        start_button = st.button("Start Camera")
        stop_button = st.button("Stop Camera")
        capture_button = st.button("Capture Frame")

        if start_button:
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                st.error("Unable to access camera")
            else:
                st.success("Camera started successfully")

            # Streamlit frame to display the live camera feed
            stframe = st.empty()

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Camera feed not available.")
                        break

                    # Convert BGR to RGB for Streamlit
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Perform detection on the live frame
                    results = model(frame)
                    for obj in results.pred[0]:
                        class_name = model.names[int(obj[5])]
                        cv2.putText(frame_rgb, f"{class_name}", (int(obj[0]), int(obj[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.rectangle(frame_rgb, (int(obj[0]), int(obj[1])), (int(obj[2]), int(obj[3])), (0, 255, 0), 2)

                    # Display the live feed with detections
                    stframe.image(frame_rgb, channels="RGB")

                    # Capture the frame when the capture button is clicked
                    if capture_button:
                        for obj in results.pred[0]:
                            class_name = model.names[int(obj[5])]
                            count = 1
                            log_detection(c, class_name, count, location)
                        st.success("Frame captured and data stored in the database.")

                    # Stop the camera if the stop button is pressed
                    if stop_button:
                        st.info("Camera stopped.")
                        break

            except Exception as e:
                st.error(f"Error: {str(e)}")
            finally:
                cap.release()
                st.info("Camera stopped.")

import streamlit as st
import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('detect.onnx', task='detect', verbose=False)

# Initialize Streamlit session state for blink count
if "blink_count" not in st.session_state:
    st.session_state.blink_count = 0
if "closed_frames" not in st.session_state:
    st.session_state.closed_frames = 0

# Function to process video frames and count blinks
def process_video(frame, model, min_closed_frames):
    frame = cv2.flip(frame, 1)  # Mirror effect
    results = model.predict(frame, show=False)

    if len(results[0].boxes.xyxy) > 0:
        cls_id = int(results[0].boxes.cls[0])

        if cls_id == 0:  # Eye is closed
            st.session_state.closed_frames += 1
        elif cls_id == 1:  # Eye is open
            if st.session_state.closed_frames >= min_closed_frames:
                st.session_state.blink_count += 1
            st.session_state.closed_frames = 0

        x1, y1, x2, y2 = map(int, results[0].boxes.xyxy[0])
        label = "Open" if cls_id == 1 else "Closed"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame

# Streamlit UI
st.title("BlinkMeter")
st.text("Using YOLO and Streamlit")

min_closed_frames = st.slider("Min Closed Frames for Blink", min_value=1, max_value=10, value=3)

# Button to start webcam
if st.button("Start Webcam"):
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        st.error("Error: Could not access the webcam.")
    else:
        stframe = st.empty()
        title_placeholder = st.empty()

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            processed_frame = process_video(frame, model, min_closed_frames)
            stframe.image(processed_frame, channels="BGR")
            title_placeholder.title(f"Blinks Detected: {st.session_state.blink_count}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()

st.text("Press 'q' to stop the application")
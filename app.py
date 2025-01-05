import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np

# Load the YOLO model
model = YOLO('best.pt', verbose=False)

# Initialize Streamlit session state for blink count
if "blink_count" not in st.session_state:
    st.session_state.blink_count = 0  # Initialize blink count
if "closed_frames" not in st.session_state:
    st.session_state.closed_frames = 0  # Initialize closed frames count

# Function to process video frames and count blinks
def process_video(frame, model, min_closed_frames):
    """
    Process a single video frame and update blink counter in session state.

    Args:
        frame: The input video frame.
        model: YOLO model instance.
        min_closed_frames: Minimum consecutive closed frames to count as a blink.
    """
    # Flip the frame for a mirror effect
    frame = cv2.flip(frame, 1)

    # Get predictions
    results = model.predict(frame, show=False)

    if len(results[0].boxes.xyxy) > 0:
        # Get the class label (0 for closed, 1 for open)
        cls_id = int(results[0].boxes.cls[0])

        # State-based blink counting
        if cls_id == 0:  # Eye is closed
            st.session_state.closed_frames += 1
        elif cls_id == 1:  # Eye is open
            if st.session_state.closed_frames >= min_closed_frames:
                st.session_state.blink_count += 1  # Count a blink
            st.session_state.closed_frames = 0  # Reset closed frames counter

        # Draw bounding box and label
        x1 = int(results[0].boxes.xyxy[0][0])
        y1 = int(results[0].boxes.xyxy[0][1])
        x2 = int(results[0].boxes.xyxy[0][2])
        y2 = int(results[0].boxes.xyxy[0][3])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
        label = "Open" if cls_id == 1 else "Closed"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame

# Streamlit UI
st.title(f"Real-Time Blink Counter")
st.text("Using YOLO and Streamlit")

# Parameters
min_closed_frames = st.slider("Min Closed Frames for Blink", min_value=1, max_value=10, value=3)
if st.button("Start Webcam"):
    # Access webcam
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        st.error("Error: Could not access the webcam.")
    else:
        stframe = st.empty()

    # Placeholder for the dynamic title
    title_placeholder = st.empty()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Process the frame
        processed_frame = process_video(frame, model, min_closed_frames)

        # Display the frame in Streamlit
        stframe.image(processed_frame, channels="BGR")

        # Update title dynamically without rendering new titles
        # Update the title dynamically
        title_placeholder.title(f"Real-Time Blink Counter - Blinks: {st.session_state.blink_count}")


        # Stop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video stream
    video_capture.release()


st.text("Press 'q' to stop the application")
import streamlit as st
import cv2
import mediapipe as mp
import time
import tensorflow as tf
import numpy as np
import math

# Page Configuration
st.set_page_config(page_title="Sign Language Recognition", page_icon="👋", layout="wide")

# Sidebar Configuration
st.sidebar.title("Sign Language Recognition")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.9, 0.05)

# Main Page Title
st.title("Sign Language Recognition System")
st.markdown("This application recognizes hand signs and translates them into text.")

# Initialize MediaPipe
@st.cache_resource
def load_mediapipe():
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.9,
        min_tracking_confidence=0.7
    )
    mpDraw = mp.solutions.drawing_utils
    return hands, mpDraw, mpHands

hands, mpDraw, mpHands = load_mediapipe()

# Load Model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("signlanguage1.h5")
    class_labels = ['AAROHAN', 'I', 'am', 'are', 'fine', 'hello', 'how', 'to', 'welcome', 'you']
    return model, class_labels

model, class_labels = load_model()

# Function to Crop Hand from Image
def crop_hand_from_image(img, handLms):
    h, w, _ = img.shape
    x_min, y_min, x_max, y_max = w, h, 0, 0

    for lm in handLms.landmark:
        x, y = int(lm.x * w), int(lm.y * h)
        x_min, y_min = min(x, x_min), min(y, y_min)
        x_max, y_max = max(x, x_max), max(y, y_max)

    # Add padding
    padding = 40
    x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
    x_max, y_max = min(w, x_max + padding), min(h, y_max + padding)

    return img[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)

# Preprocess Image for Model
def modify(imgWhite):
    if imgWhite is None or imgWhite.size == 0:
        return None
    imgGray = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2GRAY)
    imgGray = cv2.resize(imgGray, (50, 50)).astype("float32") / 255.0
    return np.expand_dims(imgGray, axis=(0, -1))

# Process Frames for Hand Detection and Recognition
def process_frame(frame, imgSize=48):
    frame = cv2.flip(frame, 1)
    imgRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRgb)
    
    prediction_text, confidence_value, hand_detected, cropped_display, processed_display = "", 0, False, None, None

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            hand_detected = True
            cropped_img, bbox = crop_hand_from_image(frame, handLms)

            if cropped_img.size == 0:
                continue

            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

            # Process for Model
            h, w = cropped_img.shape[:2]
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            aspectRatio = h / w

            if aspectRatio > 1:
                k, wCal = imgSize / h, math.ceil((imgSize / h) * w)
                imgResize = cv2.resize(cropped_img, (wCal, imgSize))
                imgWhite[:, (imgSize - wCal) // 2:(imgSize + wCal) // 2] = imgResize
            else:
                k, hCal = imgSize / w, math.ceil((imgSize / w) * h)
                imgResize = cv2.resize(cropped_img, (imgSize, hCal))
                imgWhite[(imgSize - hCal) // 2:(imgSize + hCal) // 2, :] = imgResize

            cropped_display, processed_display = cropped_img, imgWhite.copy()
            
            # Predict
            imgWhiteProcessed = modify(imgWhite)
            if imgWhiteProcessed is not None:
                prediction = model.predict(imgWhiteProcessed, verbose=0)
                confidence = np.max(prediction)
                if confidence >= confidence_threshold:
                    prediction_text, confidence_value = class_labels[np.argmax(prediction)], confidence

    return frame, prediction_text, confidence_value, hand_detected, cropped_display, processed_display

# Streamlit UI Elements
col1, col2 = st.columns([3, 1])

with col1:
    start_button, stop_button = st.button("Start"), st.button("Stop")
    video_feed, prediction_display = st.empty(), st.empty()
    confidence_bar = st.progress(0.0)

with col2:
    st.subheader("Hand Detection")
    cropped_hand_display, processed_image_display = st.empty(), st.empty()
    st.subheader("Statistics")
    fps_display = st.empty()

# Webcam Handling
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False

if start_button:
    st.session_state.camera_active = True

if stop_button:
    st.session_state.camera_active = False

# Main Loop for Webcam
if st.session_state.camera_active:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_time, fps_list = time.time(), []

    try:
        while st.session_state.camera_active:
            success, frame = cap.read()
            if not success:
                st.error("Failed to capture frame from webcam")
                break

            # Process Frame
            processed_frame, prediction, confidence, hand_detected, cropped_hand, processed_img = process_frame(frame)

            # FPS Calculation
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time, fps_list = current_time, fps_list[-9:] + [fps]

            # Display Frame & Prediction
            video_feed.image(processed_frame, channels="BGR", use_column_width=True)
            prediction_display.markdown(f"## Detected Sign: **{prediction if prediction else '...' }**")
            confidence_bar.progress(float(np.clip(confidence, 0.0, 1.0)))

            # Show FPS
            fps_display.text(f"FPS: {sum(fps_list) / len(fps_list):.1f}")

            # Display Cropped & Processed Images
            if hand_detected:
                cropped_hand_display.image(cropped_hand, channels="BGR", use_column_width=True)
                processed_image_display.image(processed_img, channels="BGR", use_column_width=True)
            else:
                cropped_hand_display.text("No hand detected")
                processed_image_display.text("No processed image")

    except Exception as e:
        st.error(f"Error: {e}")

    finally:
        cap.release()

else:
    video_feed.image(np.zeros((480, 640, 3), dtype=np.uint8), channels="BGR")

# Instructions
st.subheader("How to Use")
st.markdown("""
1. Press 'Start' to begin detection  
2. Show hand signs to the camera  
3. The predicted sign is displayed continuously  
4. Press 'Stop' to end the session  
""")

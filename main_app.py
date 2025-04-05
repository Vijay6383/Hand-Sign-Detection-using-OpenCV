import streamlit as st 
import cv2
import joblib
import numpy as np
from skimage.feature import hog
import mediapipe as mp


IMG_SIZE = (100, 100)


st.title("Real-Time Hand Sign Recognition using Random Forest")

# Load trained model
model = joblib.load("rf_model.joblib")

# Set up MediaPipe Hands and OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, static_image_mode=True, max_num_hands=1, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)
stframe = st.empty()



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture image from camera")
        break

    # Convert the frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get hand landmarks
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame (for visual feedback)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Crop and resize the hand region
            h, w, _ = frame.shape
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)
            
            # Expand the bounding box slightly for better coverage
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            cropped_image = frame[y_min:y_max, x_min:x_max]

            # Convert to grayscale and resize
            gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, IMG_SIZE)

            # Extract HOG features
            features, _ = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
            features = np.array(features).reshape(1, -1)

            # Predict digit
            prediction = model.predict(features)[0]

            # Display prediction
            cv2.putText(frame, f"Prediction: {prediction}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            stframe.image(frame, channels="BGR")

cap.release()
cv2.destroyAllWindows()
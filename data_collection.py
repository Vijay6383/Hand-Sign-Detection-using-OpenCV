import cv2
import os
import mediapipe as mp
import time

# Create directories to store images for each number (0 to 9)
for i in range(10):
    if not os.path.exists(f"hand_sign_images/train/{i}"):
        os.makedirs(f"hand_sign_images/train/{i}")

# Set up MediaPipe Hands and OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, static_image_mode=True, max_num_hands=1, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize variables
count = 0
max_images = 100  # Number of images to collect for each class
cropped_size = (100, 100)  # Desired size for cropping
current_class = None  # To store the current hand sign class

# Give user a moment to get ready
print("Starting in 5 seconds...")
time.sleep(5)

# Start capturing images for each hand sign
while count < max_images * 10:  # Collect images for each class (0-9)
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame.")
        break

    # Flip the frame horizontally for a later mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get hand landmarks
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame (for visual feedback)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            

            # Check if a hand is detected and allow user to select the number they are showing
            cv2.putText(frame, "Press key '0' to '9' for number", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow("Hand Sign", frame)

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

            # Display the croped hand
            if cropped_image.size > 0:
                cv2.imshow("cropped hand", cropped_image)

            # Capture the user input for the corresponding number (0-9)
            key = cv2.waitKey(1) & 0xFF

            # Check if user pressed a number key between 0 to 9
            if key >= ord('0') and key <= ord('9'):
                current_class = chr(key)  # Get the pressed number ('0' to '9')

                # Resize the image to a fixed size
                cropped_image_resized = cv2.resize(cropped_image, cropped_size)

                # Save the cropped image to the corresponding folder
                folder_path = f"hand_sign_images/train/{current_class}"
                image_path = os.path.join(folder_path, f"{count}.jpg")
                cv2.imwrite(image_path, cropped_image_resized)

                count += 1
                print(f"Collected {count} images for number {current_class}")

                if count >= max_images * 10:
                    print("Collected 100 images for each number (0-9).")
                    break  # Stop collecting after 100 images for each number

    # Press 'q' to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


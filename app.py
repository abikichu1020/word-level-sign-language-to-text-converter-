import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# --- 1. MediaPipe Setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# --- 2. Configuration ---
# Create 'dataset' folder if it doesn't exist
if not os.path.exists("dataset"):
    os.makedirs("dataset")

# IMPORTANT: Choose the word label you are collecting
label = "yes"  # <--- CHANGE THIS WORD for each sign you collect

cap = cv2.VideoCapture(0)
samples = 0
max_samples = 100  # number of samples (frames) per sign

# --- 3. Data Collection Loop ---
# Open the CSV file to write data
with open(f"dataset/{label}.csv", mode="w", newline="") as f:
    csv_writer = csv.writer(f)
    print(f"Starting collection for '{label}'. Hold your sign steady.")

    while samples < max_samples:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break
        
        # Flip frame horizontally for a more intuitive self-view
        frame = cv2.flip(frame, 1)
        
        # Convert the frame to RGB for MediaPipe processing
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # 1. Extract 2D (x, y) Coordinates
                coords = []
                # Append x and y for all 21 landmarks (42 elements total)
                for lm in hand_landmarks.landmark:
                    coords.append(lm.x)
                    coords.append(lm.y)
                
                # 2. Write to CSV and increment sample count
                csv_writer.writerow(coords)
                samples += 1

                # 3. Draw landmarks for visual feedback
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display status text
        cv2.putText(frame, f"Collecting {label}: {samples}/{max_samples}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the video feed
        cv2.imshow("Dataset Collection", frame)

        # Exit condition: Press 'q' to quit early
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

print(f"Finished collecting {samples} samples for '{label}'. Data saved to dataset/{label}.csv")

# --- 4. Cleanup ---
cap.release()
cv2.destroyAllWindows()
hands.close()
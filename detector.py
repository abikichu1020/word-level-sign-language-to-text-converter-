import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib # Tool for saving and loading the trained model

# --- 1. Configuration and Data Setup ---

# The directory where your CSV files are saved
DATA_DIR = "dataset" 
# Filenames (without .csv) must match your collected signs
SIGNS = ['hello', 'iloveyou', 'NO', 'OK', 'RESTROOM', 'YES'] 

# Create the label map for converting numbers (0, 1, 2, ...) back to words
label_map = {i: sign for i, sign in enumerate(SIGNS)}
print(f"Label Map: {label_map}")

# --- 2. Feature Extraction Function ---
# This must exactly match the feature extraction used during data collection (x, y coords)
def extract_features(hand_landmarks):
    coords = []
    # Only using 2D (x, y) coordinates, assuming MediaPipe's z-coord is less reliable
    for lm in hand_landmarks.landmark:
        coords.append(lm.x)
        coords.append(lm.y)
        # NOTE: If you collected (x, y, z), you must add coords.append(lm.z) here
    return np.array(coords)

# --- 3. Load Data and Train Model ---

X_data = []  # Features
y_data = []  # Labels

# Load data from all CSV files
print("Loading and preparing data...")
for label_int, sign in enumerate(SIGNS):
    try:
        file_path = os.path.join(DATA_DIR, f"{sign}.csv")
        df = pd.read_csv(file_path, header=None)
        
        # Check if the feature dimension matches (should be 42 for 21 * (x,y))
        if df.shape[1] != 42:
             print(f"Warning: {sign}.csv has {df.shape[1]} features, expected 42. Skipping.")
             continue
             
        X_data.append(df.values)
        y_data.extend([label_int] * len(df))
    except FileNotFoundError:
        print(f"Error: File not found for sign: {sign}.csv. Skipping.")

# Combine all data
X_combined = np.concatenate(X_data, axis=0)
y_combined = np.array(y_data)

# Split data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined)

# Train the Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"\n📊 Model Training Complete.")
print(f"Accuracy on Test Set: {accuracy_score(y_test, y_pred)*100:.2f}%")

# Save the trained model for faster startup next time (optional but recommended)
joblib.dump(model, 'sign_model.pkl')
print("Model saved as sign_model.pkl.")

# --- 4. Real-Time Detection and Overlay ---

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# Variables for displaying prediction
predicted_word = "No Hand"
confidence_percentage = "0%"

print("\n🚀 Starting Real-Time Detector. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for a mirror view
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to detect hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        # We only look at the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # 1. Extract Features from Live Feed
        features = extract_features(hand_landmarks)
        
        # Check if features have the correct shape (42)
        if features.size == 42:
            features = features.reshape(1, -1) 
            
            # 2. Predict Word and Confidence
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            
            # Get predicted word and its confidence
            predicted_label = prediction
            confidence = probabilities[predicted_label] * 100
            
            # Update display variables
            predicted_word = label_map[predicted_label].upper()
            confidence_percentage = f"{confidence:.0f}%"

            # 3. Draw Landmarks on the hand
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2), # Green
                                   mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)) # Red
        else:
            predicted_word = "Error"
            confidence_percentage = "0%"
                
    else:
        # Reset display if no hand is detected
        predicted_word = "No Hand"
        confidence_percentage = "0%"

    # 4. Draw Overlay Text (Matching the picture: 'word: confidence%')
    overlay_text = f"{predicted_word}: {confidence_percentage}"
    
    # Text configuration
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_thickness = 3
    text_color = (0, 255, 0)  # Green text
    bg_color = (0, 0, 0)      # Black background
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(overlay_text, font, font_scale, font_thickness)
    
    # Define position (Top left corner)
    text_x, text_y = 10, text_height + 10
    
    # Draw Black Background Rectangle
    cv2.rectangle(frame, (text_x - 5, text_y - text_height - 5), 
                  (text_x + text_width + 5, text_y + baseline + 5), bg_color, -1)
    
    # Draw Green Text
    cv2.putText(frame, overlay_text, (text_x, text_y), font, 
                font_scale, text_color, font_thickness, cv2.LINE_AA)

    # Show the final output
    cv2.imshow('Real-Time Sign Language Detection', frame)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()
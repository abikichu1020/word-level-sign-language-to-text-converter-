import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib 

# --- 1. Configuration and Data Consolidation ---

DATA_DIR = "dataset" 
# Map all collected file prefixes to a single, normalized sign word
# Use the file structure you provided to create the master list.
# NOTE: If 'HELLO.csv' and 'hello.csv' are the same sign, we map them to 'HELLO'
SIGN_MAPPING = {
    'HELLO': 'HELLO',
    'hello': 'HELLO',
    'I LOVE YOU': 'ILOVEYOU',
    'iloveyou': 'ILOVEYOU',
    'NO': 'NO',
    'NO1': 'NO',
    'OK': 'OK',
    'OKAY': 'OK',
    'RESTROOM': 'RESTROOM',
    'RESTROOM1': 'RESTROOM',
    'STOP': 'STOP',
    'STOP1': 'STOP',
    'yes': 'YES',
    'yes1': 'YES',
}

# The unique list of signs we will train on
SIGNS = sorted(list(set(SIGN_MAPPING.values()))) 
label_map = {i: sign for i, sign in enumerate(SIGNS)}
print(f"Unique Signs: {SIGNS}")
print(f"Label Map: {label_map}")

# --- 2. Feature Extraction Function ---
def extract_features(hand_landmarks):
    """Extracts 2D (x, y) coordinates from 21 landmarks into a 42-element vector."""
    coords = []
    for lm in hand_landmarks.landmark:
        coords.append(lm.x)
        coords.append(lm.y)
    return np.array(coords)

# --- 3. Load Data and Train Model (Load/Train if model doesn't exist) ---

MODEL_FILE = 'sign_model.pkl'

if os.path.exists(MODEL_FILE):
    print("🧠 Loading existing model...")
    model = joblib.load(MODEL_FILE)
else:
    print("⏳ Model file not found. Training new model...")
    X_data = []  # Features
    y_data = []  # Labels

    # Iterate through the files found in the dataset folder
    for file_name in os.listdir(DATA_DIR):
        if file_name.endswith('.csv'):
            file_prefix = file_name.replace('.csv', '')
            
            # Use the mapping to get the canonical sign name
            if file_prefix in SIGN_MAPPING:
                sign = SIGN_MAPPING[file_prefix]
                label_int = SIGNS.index(sign)

                try:
                    file_path = os.path.join(DATA_DIR, file_name)
                    df = pd.read_csv(file_path, header=None)
                    
                    if df.shape[1] == 42 and not df.empty:
                        X_data.append(df.values)
                        y_data.extend([label_int] * len(df))
                        print(f"  Loaded {len(df)} samples for '{sign}' from {file_name}")
                    else:
                        print(f"  Skipping {file_name}: Incorrect features or empty file (features={df.shape[1]})")

                except Exception as e:
                    print(f"  Error loading {file_name}: {e}")

    # Combine data
    if not X_data:
        print("❌ Error: No valid data loaded. Exiting.")
        exit()
        
    X_combined = np.concatenate(X_data, axis=0)
    y_combined = np.array(y_data)

    # Train Model
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate and Save
    y_pred = model.predict(X_test)
    print(f"\n📊 Model Training Complete. Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    joblib.dump(model, MODEL_FILE)
    print(f"Model saved as {MODEL_FILE}.")


# --- 4. Real-Time Detection and Enhanced Overlay ---

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# Variables for displaying prediction
predicted_word = "Loading..."
confidence_percentage = "0%"

print("\n🚀 Starting Real-Time Detector. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Flip the frame for a mirror view
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb_frame)

    # --- Prediction and Drawing Logic ---
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        features = extract_features(hand_landmarks)
        
        if features.size == 42:
            features = features.reshape(1, -1) 
            
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            
            predicted_label = prediction
            confidence = probabilities[predicted_label] * 100
            
            predicted_word = label_map[predicted_label].upper()
            confidence_percentage = f"{confidence:.0f}%"

            # 1. Calculate Bounding Box and Draw Enhanced Hand Outline (Attractive)
            x_coords = [lm.x * width for lm in hand_landmarks.landmark]
            y_coords = [lm.y * height for lm in hand_landmarks.landmark]
            min_x, max_x = int(min(x_coords) - 15), int(max(x_coords) + 15)
            min_y, max_y = int(min(y_coords) - 15), int(max(y_coords) + 15)
            
            # Define color based on confidence (Green for High, Red for Low)
            if confidence > 90:
                box_color = (0, 255, 0) # Bright Green
            elif confidence > 70:
                box_color = (0, 255, 255) # Yellow
            else:
                box_color = (0, 165, 255) # Orange

            # Draw a thick box around the hand (like the reference image)
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), box_color, 4) 
            
            # Draw landmarks inside the box for detail
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1), 
                                   mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1)) 
        else:
            predicted_word = "WAIT"
            confidence_percentage = "0%"
                
    else:
        predicted_word = "No Hand"
        confidence_percentage = "0%"

    # 2. Draw Overlay Text (Highly Visible Overlay like the image)
    overlay_text = f"{predicted_word}: {confidence_percentage}"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.6
    font_thickness = 4
    text_color = (255, 255, 255)  # White text
    bg_color = (0, 0, 0)          # Black background
    
    (text_width, text_height), baseline = cv2.getTextSize(overlay_text, font, font_scale, font_thickness)
    
    # Position: Top Center/Left
    text_x, text_y = 10, text_height + 25
    
    # Draw Background (Translucent Black for better visibility)
    cv2.rectangle(frame, (text_x - 15, text_y - text_height - 15), 
                  (text_x + text_width + 15, text_y + baseline + 15), bg_color, -1)
    
    # Draw Text
    cv2.putText(frame, overlay_text, (text_x, text_y), font, 
                font_scale, text_color, font_thickness, cv2.LINE_AA)

    cv2.imshow('Real-Time Sign Language Detection', frame)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()
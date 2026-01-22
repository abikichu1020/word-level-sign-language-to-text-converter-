# Word-Level Sign Language to Text Converter

## Project Title
Word-Level Sign Language to Text Converter using Machine Learning

## Repository
word-level-sign-language-to-text-converter

## Objective
The objective of this project is to recognize hand gestures representing words in sign language and convert them into readable text in real time. The system focuses on word-level classification rather than alphabet-level recognition.

## Description
This project implements a machine learning–based sign language recognition system that detects hand gestures through a webcam, extracts hand landmarks, and classifies them into predefined word categories. The predicted word is then displayed as text on the screen.

The system uses MediaPipe for hand landmark detection and a Random Forest classifier for gesture classification. It is designed for educational and assistive technology applications.

## Project Structure
- app.py : Application entry point
- detector.py : Hand detection and feature extraction logic
- detector1.py : Model training, evaluation, and real-time prediction
- dataset/ : CSV files containing landmark data for each word
  - HELLO.csv
  - ILOVEYOU.csv
  - YES.csv
  - NO.csv
  - OK.csv
  - STOP.csv
  - RESTROOM.csv
  - and variants (lower/numbered files)
- sign_model.pkl : Trained machine learning model
- requirements.txt : Python dependencies

## Technologies Used
- Python
- Machine Learning
- MediaPipe (hand landmark detection)
- OpenCV
- Scikit-learn
- Joblib

## Model Used
Random Forest Classifier

Key characteristics:
- Supervised learning
- Robust to noise
- Handles nonlinear feature relationships
- Efficient for small-to-medium datasets

## Workflow
1. Capture video from webcam
2. Detect hand landmarks using MediaPipe
3. Extract feature vectors from landmarks
4. Classify gesture using trained ML model
5. Display corresponding word as text

## Libraries Used
- mediapipe
- opencv-python
- numpy
- pandas
- scikit-learn
- joblib

## Features
- Word-level sign recognition
- Real-time prediction
- Webcam-based interaction
- Pretrained model support
- Modular code structure

## Supported Words
- HELLO
- I LOVE YOU
- YES
- NO
- OK
- STOP
- RESTROOM
(extendable by adding new datasets)

## Use Cases
- Assistive communication tools
- Human–computer interaction
- Educational demonstrations
- Accessibility applications
- Machine learning practice projects

## Advantages
- Simple and interpretable ML model
- Real-time performance
- Extendable dataset and vocabulary
- Lightweight compared to deep learning approaches

## Limitations
- Limited vocabulary
- Sensitive to hand orientation and lighting
- Requires consistent gesture execution
- Not suitable for sentence-level translation

## How to Run
1. Install Python (3.x recommended)
2. Install dependencies:
   pip install -r requirements.txt
3. Ensure webcam access is enabled
4. Run the application:
   python app.py
   or
   python detector1.py
5. Perform gestures in front of the camera

## Output
- Recognized sign displayed as text
- Real-time overlay on video feed
- Console accuracy metrics (during training)

## Conclusion
This project demonstrates a practical word-level sign language to text conversion system using traditional machine learning and computer vision techniques. It provides a solid foundation for building more advanced sign language recognition systems.

## Author
Developed as part of a machine learning / assistive technology project.

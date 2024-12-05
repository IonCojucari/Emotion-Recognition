import cv2
from mtcnn import MTCNN
import numpy as np
from keras.models import load_model
import sys
import os

# Ensure UTF-8 encoding for logs and output
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding='utf-8')

# Load the pre-trained emotion recognition model
emotion_model = load_model('Trained Models/new_data_image_model.keras')
emotion_labels = ['Angry', 'Disgust', 'Sad', 'Joy', 'Neutral', 'Scared', 'Surprised']

# Initialize MTCNN face detector
detector = MTCNN()

# Start video capture
cap = cv2.VideoCapture(0)

print("[INFO] Starting webcam feed. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Unable to capture video.")
        break
    print("[INFO] Captured frame. Processing...")

    faces = detector.detect_faces(frame)
    
    if faces:  # Process only if faces are detected
        # Process the largest face (or the first detected face)
        face = faces[0]
        x, y, width, height = face['box']
        x, y = max(0, x), max(0, y)  # Ensure coordinates are within bounds
        
        face_region = frame[y:y+height, x:x+width]  # Extract face region
        
        # Preprocess the face region
        resized_face = cv2.resize(face_region, (128, 128))  # Resize to 128x128

        # Convert the face region to grayscale and replicate across 3 channels
        gray_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        rgb_face = cv2.merge([gray_face, gray_face, gray_face])  # Replicate grayscale to 3 channels
        ##Show all the image with red square around the face
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)
        

        # Normalize the face image
        face_input = np.expand_dims(rgb_face, axis=0)  # Add batch dimension

        # Predict emotion without showing model summary
        predictions = emotion_model.predict(face_input, verbose=0)[0]
        print(predictions)
        emotion_label = emotion_labels[np.argmax(predictions)]
        
        #Show the emotion near the face
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.imshow('Webcam Feed', frame)
        


        # Display emotion text
        print(f"[INFO] Emotion: {emotion_label}")

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

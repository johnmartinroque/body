import mediapipe as mp
import cv2
import numpy as np
import joblib
import threading

import pyttsx3

# Initialize TTS engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 130)  # Adjust speech rate
last_spoken = ""  # To prevent repeating the same speech

# Load your gesture classifier
classifier = joblib.load('gesture_classifier.pkl')

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Open webcam
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
            )

        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # Extract landmarks
        landmarks = []
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])

        # Predict and speak
        if landmarks:
            landmarks = np.array(landmarks).flatten()
            try:
                prediction = classifier.predict([landmarks])[0]
                proba = classifier.predict_proba([landmarks])[0]
                confidence = np.max(proba) * 100
            except:
                prediction = "Unknown"
                confidence = 0

            # Show on screen
            cv2.putText(image, f'Gesture: {prediction} ({confidence:.2f}%)', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Speak only if confident and not already spoken
            if prediction != last_spoken and confidence > 80:
                def speak_text(text):
                    tts_engine.say(text)
                    tts_engine.runAndWait()

# Speak only if confident and not already spoken
            if prediction != last_spoken and confidence > 80:
                threading.Thread(target=speak_text, args=(prediction,), daemon=True).start()
                last_spoken = prediction
        else:
            cv2.putText(image, 'No pose or hands detected', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow('Gesture Recognition with Face Mesh', image)

        # Quit on 'q' key
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

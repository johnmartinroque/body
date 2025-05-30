import mediapipe as mp
import cv2
import numpy as np
import joblib  # For loading a trained classifier

# Load pre-trained emotion classifier (ensure you have trained and saved this)
# Example: classifier = joblib.load('emotion_classifier.pkl')
# For demo purposes, let's use a dummy classifier


classifier = joblib.load('emotion_classifier.pkl')

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # Emotion Recognition
        if results.face_landmarks:
            face = results.face_landmarks.landmark

            # Extract features (e.g., all x, y coordinates)
            face_coords = np.array([[lm.x, lm.y] for lm in face]).flatten()  # shape: (N*2,)

            # Predict emotion
            try:
                prediction = classifier.predict([face_coords])[0]
            except:
                prediction = "Unknown"

            # Display emotion
            cv2.putText(image, f'Emotion: {prediction}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Emotion Recognition Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

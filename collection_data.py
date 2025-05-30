import cv2
import mediapipe as mp
import csv
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Initialize holistic model
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

# Open CSV to save data
csv_file = open('gesture_data.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)

print("Press '1', '2', or '3' to label gestures. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)

    landmarks = []

    # Collect landmarks: pose + left/right hands
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y])
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y])
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y])

    # Draw landmarks
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    cv2.imshow('Collecting Gesture Data', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('1') and landmarks:
        csv_writer.writerow(landmarks + ['gesture1'])
        print("Saved: gesture1")
    elif key == ord('2') and landmarks:
        csv_writer.writerow(landmarks + ['gesture2'])
        print("Saved: gesture2")
    elif key == ord('3') and landmarks:
        csv_writer.writerow(landmarks + ['gesture3'])
        print("Saved: gesture3")
    elif key == ord('q'):
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import csv
import numpy as np

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1)

cap = cv2.VideoCapture(0)

# Open CSV to save data
csv_file = open('emotion_data.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)

print("Press 'h' for Happy, 's' for Sad, 'u' for sUrprised. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = []
            for lm in face_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])  # Only use x and y for simplicity

            # Draw landmarks on screen
            for lm in face_landmarks.landmark:
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)

    cv2.imshow('Collecting Data', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('h') and results.multi_face_landmarks:
        csv_writer.writerow(landmarks + ['happy'])
        print("Saved: happy")
    elif key == ord('s') and results.multi_face_landmarks:
        csv_writer.writerow(landmarks + ['sad'])
        print("Saved: sad")
    elif key == ord('u') and results.multi_face_landmarks:
        csv_writer.writerow(landmarks + ['surprised'])
        print("Saved: surprised")
    elif key == ord('q'):
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()

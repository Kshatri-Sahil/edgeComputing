import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from datetime import datetime
import os
import csv

# Load model
model = load_model("face_classifier.h5")

# Label list (order must match how ImageDataGenerator loaded them)
labels = ["Sahil", "Ravi", "Prasad"]  # Add all student names here

# Attendance dictionary
attendance_dict = {name: False for name in labels}

# CSV for attendance
def mark_attendance(name):
    if not attendance_dict[name]:
        with open('attendance.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            time_str = datetime.now().strftime('%H:%M:%S')
            date_str = datetime.now().strftime('%Y-%m-%d')
            writer.writerow([name, time_str, date_str])
        attendance_dict[name] = True
        print(f"[MARKED] {name} at {time_str}")

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam or replace with ESP32-CAM IP stream
cap = cv2.VideoCapture(0)  # or use: cv2.VideoCapture("http://<esp32-ip>:81/stream")

# Output CSV headers if file doesn’t exist
if not os.path.exists('attendance.csv'):
    with open('attendance.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Name', 'Time', 'Date'])

print("✅ Face Recognition Attendance System Started")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = face_cascade.detectMultiScale(frame, 1.1, 4)

    for (x, y, w, h) in faces:
        face_img = frame[y:y + h, x:x + w]
        face_resized = cv2.resize(face_img, (224, 224))
        face_array = np.expand_dims(preprocess_input(face_resized.astype(np.float32)), axis=0)

        # Predict
        preds = model.predict(face_array)
        label_index = np.argmax(preds[0])
        confidence = preds[0][label_index]

        # Ensure label_index is within range
        if label_index < len(labels) and confidence > 0.80:
            name = labels[label_index]
            label_text = f"{name} ({confidence*100:.1f}%)"
            color = (0, 255, 0)
            mark_attendance(name)
        else:
            name = "Unknown"
            label_text = f"{name} ({confidence*100:.1f}%)"
            color = (0, 0, 255)

        # Draw on frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Recognition Attendance", frame)

    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

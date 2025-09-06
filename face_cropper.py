import cv2
import os

# Paths
input_root = r"B:\Sem6\Attendance\students_dataset"  # your original dataset with folders: Sahil, Ravi, Prasad
output_root = r"B:\Sem6\Attendance\cropped_dataset"  # save cropped face images here

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Make output directories
if not os.path.exists(output_root):
    os.makedirs(output_root)

# Loop through each label folder
for label in os.listdir(input_root):
    input_folder = os.path.join(input_root, label)
    output_folder = os.path.join(output_root, label)
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            if len(faces) > 0:
                (x, y, w, h) = faces[0]  # first face
                face_img = img[y:y+h, x:x+w]
                resized = cv2.resize(face_img, (224, 224))  # match MobileNetV2 input
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, resized)
                print(f"[✓] Cropped: {output_path}")
            else:
                print(f"[x] No face found in {filename}")

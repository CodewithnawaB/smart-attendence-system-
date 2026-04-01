import cv2
import os
import numpy as np

dataset_path = "C:\\Face_Attendance_Project\\Dataset"

faces = []
labels = []
label_map = {}
current_label = 0

# Read dataset
for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)

    if not os.path.isdir(folder_path):
        continue

    label_map[current_label] = folder
    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        faces.append(img)
        labels.append(current_label)

    current_label += 1

labels = np.array(labels)

# Train LBPH model
model = cv2.face.LBPHFaceRecognizer_create()
model.train(faces, labels)

# Save model
model.save("face_model.yml")

print("✅ Model trained successfully!")
print("📌 Label Map:", label_map)
print("💾 Model saved as 'face_model.yml'")
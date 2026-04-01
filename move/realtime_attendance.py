import cv2
import numpy as np
import joblib
from keras_facenet import FaceNet
from datetime import datetime
import csv
import os

# =============================
# FILES
# =============================
SEEN_FILE = "seen_today.csv"

# Create seen file if not exists
if not os.path.exists(SEEN_FILE):
    with open(SEEN_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Roll_No", "Name", "Time", "Date"])

# =============================
# LOAD MODELS
# =============================
embedder = FaceNet()
classifier = joblib.load(
    "C:\\Face_Attendance_Project\\face_classifier.pkl"
)

# =============================
# CAMERA SETUP
# =============================
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

marked = set()

# =============================
# REAL-TIME LOOP
# =============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (160, 160))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        emb = embedder.embeddings([face])[0]
        name = classifier.predict([emb])[0]

        # Mark attendance ONCE per student
        if name not in marked:
            now = datetime.now()
            with open(SEEN_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    name,
                    name,
                    now.strftime("%H:%M:%S"),
                    now.strftime("%Y-%m-%d")
                ])
            marked.add(name)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            name,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =============================
# CLEANUP
# =============================
cap.release()
cv2.destroyAllWindows()
print("✅ Real-time attendance captured successfully!")
print(f"💾 Attendance saved in '{SEEN_FILE}'")
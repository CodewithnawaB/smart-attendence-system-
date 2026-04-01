import cv2
import os

video_path = "C:\\Face_Attendance_Project\\videos\\232541_tufail_khanzada.mp4\\WhatsApp Video 2026-01-12 at 12.41.38 PM (1).mp4"
save_dir = "C:\\Face_Attendance_Project\\Dataset\\232541_tufail_khanzada"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

count = 0
frame_skip = 5 # save every 5th frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (160, 160))
        cv2.imwrite(f"{save_dir}/img_{count}.jpg", face)
        count += 1

    if count >=100:  # limit images
        break

cap.release()
print("Face images extracted successfully!")

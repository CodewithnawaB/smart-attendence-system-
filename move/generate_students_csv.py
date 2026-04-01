import os
import csv

DATASET_DIR = "C:\\Face_Attendance_Project\\Dataset"
OUTPUT_FILE = "students.csv"

students = []

for folder in os.listdir(DATASET_DIR):
    folder_path = os.path.join(DATASET_DIR, folder)
    if os.path.isdir(folder_path):
        roll_no = folder
        name = folder.replace("_", " ").title()
        students.append([roll_no, name])

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Roll_No", "Name"])
    writer.writerows(students)

print("✅ students.csv generated automatically!")

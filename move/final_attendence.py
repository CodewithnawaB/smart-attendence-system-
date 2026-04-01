import csv
from datetime import date

TODAY = str(date.today())

# Load all students
students = {}
with open("C:\\Face_Attendance_Project\\students.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Split combined Roll_No and name if needed
        roll_name = row["Roll_No"]
        roll_no = roll_name.split("_", 1)[0].strip()          # "232506"
        name_part = roll_name.split("_", 1)[1] if "_" in roll_name else row["Name"]
        student_name = " ".join(name_part.split("_")).title()  # "Muhammad Ishfaq"
        students[roll_no] = student_name

# Load seen students
seen = set()
with open("C:\\Face_Attendance_Project\\seen_today.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Normalize seen roll numbers
        seen_roll = row["Roll_No"].split("_", 1)[0].strip()
        seen.add(seen_roll)

# Generate final attendance
with open("final_attendance.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    
    # Write header first
    writer.writerow(["Roll_No", "Name", "Date", "Status"])
    
    # Write attendance
    for roll_no, student_name in students.items():
        status = "Present" if roll_no in seen else "Absent"
        writer.writerow([roll_no, student_name, TODAY, status])

print("✅ Final attendance generated correctly")
print("💾 Final attendance saved as 'final_attendance.csv'")

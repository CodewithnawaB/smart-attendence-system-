import os
import ssl
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context

import cv2
import numpy as np
import pickle
import pandas as pd
from datetime import datetime
from collections import defaultdict
from keras_facenet import FaceNet
from sklearn.neighbors import KNeighborsClassifier
import csv

class SmartAttendanceSystem:
    def __init__(self, embeddings_path='C:\\Face_Attendance_Project\\embeddings.npy', 
                 labels_path='C:\\Face_Attendance_Project\\labels.npy',
                 student_db_path='C:\\Face_Attendance_Project\\students.csv'):
        """Initialize the attendance system with FaceNet"""
        
        print("Loading FaceNet embedder...")
        try:
            self.embedder = FaceNet()
            print("✓ FaceNet loaded!")
        except Exception as e:
            print(f"❌ Error loading FaceNet: {e}")
            raise
        
        # Load embeddings and labels
        print("Loading embeddings and labels...")
        try:
            self.embeddings = np.load(embeddings_path)
            self.labels = np.load(labels_path, allow_pickle=True)
            print(f"✓ Loaded {len(self.embeddings)} embeddings")
        except Exception as e:
            print(f"❌ Error loading embeddings: {e}")
            raise
        
        # Train KNN classifier
        print("Training KNN classifier...")
        try:
            self.classifier = KNeighborsClassifier(n_neighbors=5)
            self.classifier.fit(self.embeddings, self.labels)
            print("✓ Classifier trained!")
        except Exception as e:
            print(f"❌ Error training classifier: {e}")
            raise
        
        # Load student database
        print("Loading student database...")
        self.student_db = {}
        try:
            if os.path.exists(student_db_path):
                with open(student_db_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        self.student_db[row['Roll_No']] = {
                            'Name': row['Name'],
                            'Class': row.get('Class', ''),
                            'Section': row.get('Section', '')
                        }
                print(f"✓ Loaded {len(self.student_db)} students from database")
            else:
                print(f"⚠️ Student database not found at {student_db_path}")
                print("   Run: python student_database_setup.py")
        except Exception as e:
            print(f"⚠️ Error loading student database: {e}")
        
        # Face detection cascade
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Attendance tracking
        self.marked_today = defaultdict(lambda: {'time': None, 'count': 0})
        self.attendance_data = []
        
        # CSV files
        today = datetime.now().strftime("%Y-%m-%d")
        self.daily_csv = f'attendance_{today}.csv'
        self.final_csv = f'final_attendance_{today}.csv'
        
        print("✓ System initialized successfully!\n")
    
    def get_student_name(self, roll_no):
        """Get student name from roll number"""
        if roll_no in self.student_db:
            return self.student_db[roll_no]['Name']
        return roll_no  # Fallback to roll_no if not found
    
    def get_embedding(self, face_img):
        """Get face embedding using FaceNet"""
        try:
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_rgb = cv2.resize(face_rgb, (160, 160))
            embedding = self.embedder.embeddings([face_rgb])[0]
            return embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None
    
    def recognize_face(self, embedding, confidence_threshold=0.8):
        """Recognize face using KNN classifier"""
        try:
            if embedding is None:
                return "Unknown", 1.0
            
            distances, indices = self.classifier.kneighbors([embedding], n_neighbors=1)
            distance = distances[0][0]
            
            if distance < confidence_threshold:
                person_id = self.labels[indices[0][0]]
                return person_id, distance
            else:
                return "Unknown", distance
        except Exception as e:
            print(f"Error recognizing face: {e}")
            return "Unknown", 1.0
    
    def log_attendance(self, roll_no, status="Present"):
        """Log attendance with duplicate prevention"""
        current_time = datetime.now()
        
        # Check if already marked in last 30 seconds
        if roll_no in self.marked_today:
            last_time = self.marked_today[roll_no]['time']
            if last_time and (current_time - last_time).seconds < 30:
                self.marked_today[roll_no]['count'] += 1
                return False  # Duplicate
        
        # Get student name
        student_name = self.get_student_name(roll_no)
        
        # Log attendance
        timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
        self.marked_today[roll_no]['time'] = current_time
        
        # Add to attendance data
        record = {
            'Roll_No': roll_no,
            'Name': student_name,
            'Time': timestamp.split()[1],
            'Date': timestamp.split()[0],
            'Status': status
        }
        self.attendance_data.append(record)
        
        # Save immediately
        self.save_daily_attendance()
        
        return True
    
    def save_daily_attendance(self):
        """Save attendance to daily CSV"""
        if self.attendance_data:
            df = pd.DataFrame(self.attendance_data)
            # Keep only first occurrence per student per day
            df = df.drop_duplicates(subset=['Roll_No', 'Date'], keep='first')
            df.to_csv(self.daily_csv, index=False)
    
    def generate_final_report(self):
        """Generate final attendance report with all students"""
        if os.path.exists(self.daily_csv):
            df = pd.read_csv(self.daily_csv)
        else:
            df = pd.DataFrame(self.attendance_data)
        
        # Ensure proper column order
        df = df[['Roll_No', 'Name', 'Time', 'Date', 'Status']]
        
        # Add absent students
        all_roll_nos = set(self.student_db.keys())
        present_roll_nos = set(df['Roll_No'].unique())
        absent_roll_nos = all_roll_nos - present_roll_nos
        
        if absent_roll_nos:
            today = datetime.now().strftime("%Y-%m-%d")
            absent_records = []
            for roll_no in absent_roll_nos:
                absent_records.append({
                    'Roll_No': roll_no,
                    'Name': self.get_student_name(roll_no),
                    'Time': '-',
                    'Date': today,
                    'Status': 'Absent'
                })
            
            absent_df = pd.DataFrame(absent_records)
            df = pd.concat([df, absent_df], ignore_index=True)
        
        # Sort by Roll_No
        df = df.sort_values('Roll_No').reset_index(drop=True)
        
        # Save final report
        df.to_csv(self.final_csv, index=False)
        
        print(f"\n✓ Final report saved: {self.final_csv}")
        print("\n" + df.to_string(index=False))
        
        return df
    
    def run_realtime(self, duration_minutes=None):
        """Run real-time attendance system"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print("❌ Cannot open camera!")
            return
        
        start_time = datetime.now()
        print("\n🎥 Starting Real-Time Attendance System...")
        print("Press 'q' to stop | Press 's' to save report\n")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame from camera")
                break
            
            frame_count += 1
            
            # Check duration limit
            if duration_minutes and (datetime.now() - start_time).seconds > duration_minutes * 60:
                print(f"\n⏱️ Duration limit ({duration_minutes}min) reached")
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))
            
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                
                try:
                    embedding = self.get_embedding(face_img)
                    
                    if embedding is not None:
                        person_id, distance = self.recognize_face(embedding, confidence_threshold=0.8)
                        
                        if person_id != "Unknown":
                            logged = self.log_attendance(person_id, status="Present")
                            color = (0, 255, 0)
                            student_name = self.get_student_name(person_id)
                            status = "✓ MARKED" if logged else "⚠ DUP"
                            label = f"{student_name} {status}"
                            print(f"[{frame_count}] {person_id}: {student_name} | {status} (Dist: {distance:.3f})")
                        else:
                            color = (0, 0, 255)
                            label = "Unknown"
                        
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(frame, label, (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        cv2.putText(frame, f"D: {distance:.3f}", (x, y+h+20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                except Exception as e:
                    print(f"Error processing face: {e}")
            
            # Display stats
            cv2.putText(frame, f"Present: {len(self.marked_today)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Time: {datetime.now().strftime('%H:%M:%S')}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Frame: {frame_count} | Faces: {len(faces)}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.imshow('Smart Attendance System', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n⛔ Stopping system...")
                break
            elif key == ord('s'):
                self.generate_final_report()
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Final report
        print("\n📊 ATTENDANCE SUMMARY")
        print("=" * 70)
        print(f"Total frames processed: {frame_count}")
        print(f"Total students present: {len(self.marked_today)}")
        print("-" * 70)
        for roll_no, data in sorted(self.marked_today.items()):
            student_name = self.get_student_name(roll_no)
            time_str = data['time'].strftime("%H:%M:%S") if data['time'] else "N/A"
            print(f"✓ {roll_no:10} | {student_name:25} | {time_str}")
        print("=" * 70)
        
        # Auto-save final report
        self.generate_final_report()


# MAIN USAGE
if __name__ == "__main__":
    try:
        system = SmartAttendanceSystem(
            embeddings_path='C:\\Face_Attendance_Project\\embeddings.npy',
            labels_path='C:\\Face_Attendance_Project\\labels.npy',
            student_db_path='C:\\Face_Attendance_Project\\students.csv'
        )
        
        # Run for 60 minutes (or until manually stopped with 'q')
        system.run_realtime(duration_minutes=60)
        
    except FileNotFoundError as e:
        print(f"\n❌ FILE NOT FOUND ERROR: {e}")
        print("\nMake sure these files exist in: C:\\Face_Attendance_Project\\")
        print("  ✓ embeddings.npy")
        print("  ✓ labels.npy")
        print("  ✓ students.csv")
        print("\nTo create students.csv, run:")
        print("  python student_database_setup.py")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
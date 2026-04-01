import csv
import os

class StudentDatabase:
    """Create and manage student database"""
    
    def __init__(self, database_file='C:\\Face_Attendance_Project\\students.csv'):
        self.database_file = database_file
        self.students = {}
        self.load_or_create_database()
    
    def load_or_create_database(self):
        """Load existing database or create new one"""
        if os.path.exists(self.database_file):
            print(f"Loading existing database: {self.database_file}")
            with open(self.database_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.students[row['Roll_No']] = {
                        'Name': row['Name'],
                        'Class': row.get('Class', ''),
                        'Section': row.get('Section', '')
                    }
            print(f"✓ Loaded {len(self.students)} students\n")
        else:
            print("Creating new student database...")
            self.create_default_database()
    
    def create_default_database(self):
        """Create database from your existing students"""
        # Extract student info from your folder names
        dataset_path = 'C:\\Face_Attendance_Project\\Dataset'
        
        students_data = []
        
        # Map folder names to actual student info
        student_mapping = {
            '232506_muhammad_ishfaq': {
                'Roll_No': '232506',
                'Name': 'Muhammad Ishfaq',
                'Class': 'BS-AI',
                'Section': 'A'
            },
            '232512_Shahid_ali': {
                'Roll_No': '232512',
                'Name': 'Shahid Ali',
                'Class': 'BS-AI',
                'Section': 'A'
            },
            '232520_Malak_abdul_aziz': {
                'Roll_No': '232520',
                'Name': 'Malak Abdul Aziz',
                'Class': 'BS-AI',
                'Section': 'A'
            },
            '232530_Muhammad_shahab': {
                'Roll_No': '232530',
                'Name': 'Muhammad Shahab',
                'Class': 'BS-AI',
                'Section': 'A'
            },
            '232541_tufail_khanzada': {
                'Roll_No': '232541',
                'Name': 'Tufail Khanzada',
                'Class': 'BS-AI',
                'Section': 'A'
            }
        }
        
        # Check which folders exist
        for folder_name, student_info in student_mapping.items():
            folder_path = os.path.join(dataset_path, folder_name)
            if os.path.isdir(folder_path):
                students_data.append(student_info)
                self.students[student_info['Roll_No']] = {
                    'Name': student_info['Name'],
                    'Class': student_info['Class'],
                    'Section': student_info['Section']
                }
        
        # Save to CSV
        with open(self.database_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['Roll_No', 'Name', 'Class', 'Section'])
            writer.writeheader()
            writer.writerows(students_data)
        
        print(f"✓ Created student database with {len(students_data)} students")
        print(f"✓ Saved to: {self.database_file}\n")
        
        # Display
        print("Student Database:")
        print("="*60)
        for row in students_data:
            print(f"Roll: {row['Roll_No']} | Name: {row['Name']} | Class: {row['Class']}")
        print("="*60 + "\n")
    
    def get_student_name(self, roll_no):
        """Get student name from roll number"""
        if roll_no in self.students:
            return self.students[roll_no]['Name']
        return "Unknown"
    
    def get_all_roll_nos(self):
        """Get list of all roll numbers"""
        return list(self.students.keys())
    
    def add_student(self, roll_no, name, class_name='', section=''):
        """Add new student"""
        self.students[roll_no] = {
            'Name': name,
            'Class': class_name,
            'Section': section
        }
        self.save_database()
    
    def save_database(self):
        """Save database to CSV"""
        with open(self.database_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['Roll_No', 'Name', 'Class', 'Section'])
            writer.writeheader()
            for roll_no, info in self.students.items():
                writer.writerow({
                    'Roll_No': roll_no,
                    'Name': info['Name'],
                    'Class': info.get('Class', ''),
                    'Section': info.get('Section', '')
                })
        print(f"✓ Database saved to {self.database_file}")


if __name__ == "__main__":
    # Create/load database
    db = StudentDatabase()
    
    # Show all students
    print("\nAll Students in Database:")
    for roll_no in db.get_all_roll_nos():
        name = db.get_student_name(roll_no)
        print(f"  {roll_no}: {name}")
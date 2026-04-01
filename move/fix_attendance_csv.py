import pandas as pd
import os
from datetime import datetime

class FixAttendanceCSV:
    """Fix and clean attendance CSV files"""
    
    def __init__(self, attendance_file='attendance_2026-01-10.csv',
                 final_file='final_attendance.csv'):
        self.attendance_file = attendance_file
        self.final_file = final_file
    
    def clean_csv(self, input_file, output_file):
        """Clean and fix CSV file"""
        
        if not os.path.exists(input_file):
            print(f"❌ File not found: {input_file}")
            return False
        
        try:
            # Read the file
            df = pd.read_csv(input_file)
            
            print(f"\nOriginal columns: {list(df.columns)}")
            print(f"Original shape: {df.shape}")
            print("\nFirst few rows (BEFORE):")
            print(df.to_string())
            
            # Clean the data
            df_clean = pd.DataFrame()
            
            # Fix Roll_No column
            if 'Roll_No' in df.columns:
                # Extract just the number from strings like "232506_muhammad_ishfaq"
                df_clean['Roll_No'] = df['Roll_No'].astype(str).str.extract(r'(\d+)', expand=False)
            
            # Fix Name column
            if 'Name' in df.columns:
                df_clean['Name'] = df['Name'].astype(str).str.strip()
            
            # Fix Time column
            if 'Time' in df.columns:
                df_clean['Time'] = df['Time'].astype(str).str.strip()
                # Replace '-' with empty for absent students
                df_clean['Time'] = df_clean['Time'].replace('-', '')
            else:
                df_clean['Time'] = ''
            
            # Fix Date column
            if 'Date' in df.columns:
                df_clean['Date'] = df['Date'].astype(str).str.strip()
            else:
                df_clean['Date'] = datetime.now().strftime("%Y-%m-%d")
            
            # Fix Status column
            if 'Status' in df.columns:
                df_clean['Status'] = df['Status'].astype(str).str.strip()
            else:
                df_clean['Status'] = 'Unknown'
            
            # Determine Present/Absent based on Time
            df_clean['Status'] = df_clean.apply(
                lambda row: 'Present' if row['Time'] and row['Time'] != '' else 'Absent',
                axis=1
            )
            
            # Keep only needed columns in correct order
            df_clean = df_clean[['Roll_No', 'Name', 'Time', 'Date', 'Status']]
            
            # Remove duplicates (keep first occurrence)
            df_clean = df_clean.drop_duplicates(subset=['Roll_No', 'Date'], keep='first')
            
            # Sort by Roll_No
            df_clean['Roll_No'] = pd.to_numeric(df_clean['Roll_No'], errors='coerce')
            df_clean = df_clean.sort_values('Roll_No').reset_index(drop=True)
            df_clean['Roll_No'] = df_clean['Roll_No'].astype(int).astype(str)
            
            # Save cleaned file
            df_clean.to_csv(output_file, index=False)
            
            print(f"\n✅ Cleaned columns: {list(df_clean.columns)}")
            print(f"✅ Cleaned shape: {df_clean.shape}")
            print("\nFirst few rows (AFTER):")
            print(df_clean.to_string(index=False))
            
            print(f"\n✓ Saved cleaned file: {output_file}\n")
            
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def add_all_students(self, attendance_file, all_students_file, output_file):
        """Add absent students to final report"""
        
        try:
            # Read attendance file
            df_present = pd.read_csv(attendance_file)
            
            # Read all students list
            df_all = pd.read_csv(all_students_file)
            
            # Find present students
            present_rolls = set(df_present['Roll_No'].astype(str).values)
            
            # Find absent students
            all_rolls = set(df_all['Roll_No'].astype(str).values)
            absent_rolls = all_rolls - present_rolls
            
            # Create absent records
            absent_records = []
            today = datetime.now().strftime("%Y-%m-%d")
            
            for roll_no in absent_rolls:
                student_info = df_all[df_all['Roll_No'].astype(str) == roll_no].iloc[0]
                absent_records.append({
                    'Roll_No': str(roll_no),
                    'Name': student_info['Name'],
                    'Time': '',
                    'Date': today,
                    'Status': 'Absent'
                })
            
            # Combine present and absent
            df_absent = pd.DataFrame(absent_records)
            df_final = pd.concat([df_present, df_absent], ignore_index=True)
            
            # Sort by Roll_No
            df_final['Roll_No'] = df_final['Roll_No'].astype(int)
            df_final = df_final.sort_values('Roll_No').reset_index(drop=True)
            df_final['Roll_No'] = df_final['Roll_No'].astype(str)
            
            # Save final file
            df_final.to_csv(output_file, index=False)
            
            print(f"\n✓ Final report with all students: {output_file}")
            print(df_final.to_string(index=False))
            
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False


# MAIN USAGE
if __name__ == "__main__":
    fixer = FixAttendanceCSV()
    
    print("="*70)
    print("ATTENDANCE CSV FIXER")
    print("="*70)
    
    # Fix current attendance file
    print("\n1️⃣ CLEANING: attendance_2026-01-10.csv")
    print("-"*70)
    fixer.clean_csv(
        'attendance_2026-01-10.csv',
        'attendance_2026-01-10_FIXED.csv'
    )
    
    # Fix final attendance file
    print("\n2️⃣ CLEANING: final_attendance.csv")
    print("-"*70)
    fixer.clean_csv(
        'final_attendance.csv',
        'final_attendance_FIXED.csv'
    )
    
    # Create students.csv if needed
    print("\n3️⃣ CREATING: students.csv (All students database)")
    print("-"*70)
    students_data = {
        'Roll_No': ['232506', '232512', '232520', '232530'],
        'Name': ['Muhammad Ishfaq', 'Shahid Ali', 'Malak Abdul Aziz', 'Muhammad Shahab'],
        'Class': ['BS-AI', 'BS-AI', 'BS-AI', 'BS-AI'],
        'Section': ['A', 'A', 'A', 'A']
    }
    df_students = pd.DataFrame(students_data)
    df_students.to_csv('students.csv', index=False)
    print("✓ Created: students.csv")
    print(df_students.to_string(index=False))
    
    # Create final report with absent students
    print("\n4️⃣ CREATING: final_attendance_with_all_students.csv")
    print("-"*70)
    fixer.add_all_students(
        'attendance_2026-01-10_FIXED.csv',
        'students.csv',
        'final_attendance_with_all_students_FIXED.csv'
    )
    
    print("\n" + "="*70)
    print("✅ PROCESS COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  ✓ attendance_2026-01-10_FIXED.csv")
    print("  ✓ final_attendance_FIXED.csv")
    print("  ✓ students.csv")
    print("  ✓ final_attendance_with_all_students_FIXED.csv")
    print("\n" + "="*70)
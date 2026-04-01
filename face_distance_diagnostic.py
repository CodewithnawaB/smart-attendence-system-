import os
import ssl
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context

import cv2
import numpy as np
from keras_facenet import FaceNet
from sklearn.neighbors import KNeighborsClassifier

class FaceDistanceDiagnostic:
    """Diagnose why faces aren't being recognized"""
    
    def __init__(self, embeddings_path='C:\\Face_Attendance_Project\\embeddings.npy',
                 labels_path='C:\\Face_Attendance_Project\\labels.npy'):
        
        print("Loading FaceNet...")
        self.embedder = FaceNet()
        
        print("Loading embeddings...")
        self.embeddings = np.load(embeddings_path)
        self.labels = np.load(labels_path, allow_pickle=True)
        
        # Train classifier
        self.classifier = KNeighborsClassifier(n_neighbors=5)
        self.classifier.fit(self.embeddings, self.labels)
        
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        print(f"✓ Loaded {len(self.embeddings)} embeddings for {len(np.unique(self.labels))} students\n")
    
    def get_embedding(self, face_img):
        """Get face embedding"""
        try:
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_rgb = cv2.resize(face_rgb, (160, 160))
            embedding = self.embedder.embeddings([face_rgb])[0]
            return embedding
        except:
            return None
    
    def run_diagnostic(self):
        """Run real-time diagnostic"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("🎥 Starting Face Distance Diagnostic...")
        print("Show your face to camera and watch the distances")
        print("Press 'q' to quit\n")
        
        distances_log = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))
            
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                
                try:
                    embedding = self.get_embedding(face_img)
                    
                    if embedding is not None:
                        # Get all distances to training samples
                        distances, indices = self.classifier.kneighbors([embedding], n_neighbors=5)
                        distances = distances[0]
                        
                        # Get student names for each neighbor
                        neighbor_names = [self.labels[idx] for idx in indices[0]]
                        
                        # Display on frame
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        # Show top 5 distances
                        y_offset = y - 10
                        cv2.putText(frame, "Top 5 Matches:", (x, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        y_offset -= 30
                        
                        for i, (dist, name) in enumerate(zip(distances, neighbor_names)):
                            color = (0, 255, 0) if dist < 0.6 else (0, 165, 255) if dist < 1.0 else (0, 0, 255)
                            text = f"{i+1}. {name}: {dist:.3f}"
                            cv2.putText(frame, text, (x, y_offset),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                            y_offset -= 25
                            
                            distances_log.append({
                                'name': name,
                                'distance': dist,
                                'threshold_passed': dist < 0.6
                            })
                
                except Exception as e:
                    print(f"Error: {e}")
            
            cv2.putText(frame, "Distances < 0.6 = MATCH (Green)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Distances 0.6-1.0 = MAYBE (Orange)", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            cv2.putText(frame, "Distances > 1.0 = NO MATCH (Red)", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Face Distance Diagnostic', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Print statistics
        print("\n" + "="*60)
        print("DIAGNOSTIC RESULTS")
        print("="*60)
        
        if distances_log:
            import pandas as pd
            df = pd.DataFrame(distances_log)
            
            print("\nDistance Statistics:")
            print(df.groupby('name')['distance'].describe())
            
            print("\nMatches at different thresholds:")
            for threshold in [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
                matches = len(df[df['distance'] < threshold])
                percentage = (matches / len(df) * 100) if len(df) > 0 else 0
                print(f"  Threshold {threshold}: {matches}/{len(df)} matches ({percentage:.1f}%)")
            
            print("\n✅ RECOMMENDATION:")
            avg_dist = df['distance'].mean()
            if avg_dist < 0.5:
                print(f"   Current threshold (0.6) is GOOD - Avg distance: {avg_dist:.3f}")
            elif avg_dist < 0.7:
                print(f"   INCREASE threshold to 0.7 - Avg distance: {avg_dist:.3f}")
            elif avg_dist < 1.0:
                print(f"   INCREASE threshold to 1.0 - Avg distance: {avg_dist:.3f}")
            else:
                print(f"   ⚠️ Distances too high ({avg_dist:.3f}) - Check image quality/lighting")
        else:
            print("No faces detected!")
        
        print("="*60 + "\n")


if __name__ == "__main__":
    try:
        diagnostic = FaceDistanceDiagnostic()
        diagnostic.run_diagnostic()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()